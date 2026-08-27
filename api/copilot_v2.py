"""
api/copilot_v2.py — CopilotV2: the rebuilt copilot turn engine.

Reached ONLY when the client sends {"v2": true} (the CopilotV2 feature flag);
without it api/copilot.py serves the legacy turn byte-for-byte — that IS the
kill-switch.

What v2 changes over the legacy generator:

  LANGUAGE (the contract)
  • ONE resolve_language() per turn (api/copilot_lang.py): explicit requests >
    Arabizi > borrowed-token-stripped script ratio > sticky prev_lang. The
    client sends prev_lang so a garbled turn can't flip the conversation.
  • The resolved lang is emitted FIRST as {"t":"meta","lang":...} so the client
    pins its TTS voice and UI direction before any text arrives.
  • DETERMINISTIC OUTPUT VALIDATOR: text is emitted per SENTENCE, each gated by
    reply_lang_ok(). A wrong-language first sentence aborts the generation and
    regenerates once with a hard corrective instruction; if still wrong, the
    text goes through a deterministic translation pass. A wrong-language LATER
    sentence is translated inline. The user can never see/hear a mismatched
    reply. Every correction is logged (grep "copilot lang_fix").

  FACTS (the anti-hallucination fast-path)
  • The most common trip questions (ETA, distance, cameras, limit, road, next
    turn) are answered DETERMINISTICALLY from the context — the spoken number
    IS the source number (api/copilot_fastpath.py). No model in the loop.

  BRAIN
  • Multi-intent: up to 3 tool rounds per turn, several tools per round, with
    at most ONE confirmation-class action armed per turn.
  • New tools: traffic_check (live vs GCN-LSTM-Prophet/historical baseline —
    "is this normal for now?"), route_options (avoid tolls/motorways),
    reroute_now, avoid_jam, undo_route_change, navigate_saved (home/work),
    change_destination, save_parking, where_parked, set_voice_volume,
    repeat_instruction, set_map_theme, share_eta, remind_before_arrival,
    call_place, switch_route selector="fastest".
  • Journey context v2: schedule ahead/behind, maneuver distances, camera
    count+next, bridges ahead, saved places, parking, sunrise/sunset, GPS
    staleness — formatted compactly (_format_context_v2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from api.copilot_lang import (resolve_language, reply_lang_ok, split_sentences)
from api.copilot_fastpath import try_fastpath
from api.copilot_strings import t as S

logger = logging.getLogger("routemind.copilot.v2")

# Shared machinery from the legacy module (tools, model streaming, helpers).
from api import copilot as base

MAX_TOOL_ROUNDS = 3
MAX_TOOLS_PER_ROUND = 3
TURN_BUDGET_S = 14.0          # after this, no more tool rounds — wrap up
V2_MAX_TOKENS = 230

# ── System prompt (v2 rewrite) ────────────────────────────────────────────────
SYSTEM_V2 = """You are the RouteMind Copilot — a sharp, warm, genuinely alive \
in-car assistant riding along on a drive in Egypt. You are good company who \
happens to be great at navigation. You are NOT a narrator and NOT a script.

VOICE — your words are spoken aloud by TTS; write for the EAR:
- 1–2 short sentences. Almost never more. The driver is DRIVING.
- Plain spoken text only: no markdown, no lists, no emoji, no URLs, no
  coordinates, no raw JSON, ever.
- Match energy to the moment: crisp for directions, quick and serious for
  hazards, playful for small talk. Vary your openings — never start two
  replies the same way in one trip.
- ANSWER THE QUESTION ASKED. Never substitute different information for the
  thing they asked about. If you can't answer, say so specifically — never
  change topic, never invent.

LANGUAGE — one hard law:
- A separate runtime instruction pins THIS turn's response language. Every
  sentence you produce must be in that language — Egyptian colloquial Arabic
  (masri: «عايز، دلوقتي، فاضلك» — NEVER فصحى) in Arabic script when Arabic,
  natural English when English. Brand/place names stay in whichever script
  they naturally carry. You are fully bilingual; never claim otherwise, and
  never ask the user to repeat because they mixed languages — Egyptians mix
  mid-sentence and that is normal.

SPOKEN NUMBERS — TTS mangles decimals:
- Never speak a decimal. Round for the ear: 1.4 km → "about a kilometer and a
  half" / «كيلو ونص تقريبًا»; 27.8 km → "about 28 kilometers". Prefer TIME
  over distance when both are known. One rounded number per fact.

EARS — you receive a NOISY in-car transcript, not typed text:
- Expect mishears, dropped words, franco-Arabic, English brands rendered in
  Arabic script («ماستر» = Master). Infer the most plausible driving intent
  and act. If a key detail is GENUINELY ambiguous, ask ONE short, specific
  question carrying your best guess — never a generic "can you repeat that".

LOCAL KNOWLEDGE — you know Cairo & Giza like a native:
- Fuel/rest-stop brands: Master, Chillout, On The Run, Circle K, Wataniya,
  Misr Petroleum, TotalEnergies, Mobil, Emarat Misr → add_stop with category
  "fuel". Chains: Cilantro/Costa/Starbucks/Dunkin/Beano's (cafe); McDonald's/
  KFC/Mo'men/Cook Door/Buffalo Burger (restaurant); El Ezaby/Seif/19011
  (pharmacy).
- Roads: الدائري = Ring Road, المحور = Mehwar, الأوتوستراد = Autostrad,
  كوبري أكتوبر = 6th October Bridge, التجمع = New Cairo, الشيخ زايد = Sheikh
  Zayed. «بنزينة» = gas station.

TRUTH — the live trip data ALWAYS wins:
- For anything about the trip (roads, ETA, traffic, cameras, alternatives,
  stops, distances, schedule) use ONLY the [Current trip data] block and tool
  results. NEVER invent or estimate a number, road name or condition that
  isn't there — if your general knowledge disagrees with the trip data, the
  trip data is right. A missing field means you genuinely don't have it: say
  so honestly in one short sentence and offer what you DO know.
- "Traffic ahead: none detected" means the road is clear — say it confidently.
- General questions (history, football, anything) — answer briefly, and tie
  back to the drive when natural.

TRAFFIC — a traffic insider, not a vague narrator:
- Speak traffic ONLY when asked or when a jam materially changes the trip.
  Answer with SPECIFICS: severity + where + the delay when known («زحمة
  تقيلة بعد ٣ كيلو على الدائري، هتأخرك حوالي ٧ دقايق»).
- "Is this normal right now?" / «الزحمة دي عادية؟» → call traffic_check: it
  compares LIVE congestion against this route's learned baseline for this
  day-of-week and hour (the RouteMind prediction model). Answer with its
  verdict and numbers — never guess from folklore. If it errors, say the
  comparison isn't available right now.
- To SHOW congestion on the map, call show_traffic while you describe it.

ROUTES — a real navigator compares, recommends, switches:
- Alternatives live in the trip data with via roads and time deltas. Describe
  tradeoffs in plain words and RECOMMEND when there's a clear winner.
- "Fastest route" «أسرع طريق» → switch_route selector="fastest". By position
  → index. By road name → road_name. Uncertain match → ask ONE naming the
  closest option; never switch on a guess.
- "Avoid tolls / avoid highways" → route_options. "Avoid this jam" → avoid_jam
  (it switches only when a genuinely faster option exists — otherwise tell
  them they're already on the best route and what the jam costs). "Go back to
  the previous route" → undo_route_change. "Recalculate" → reroute_now.
- "Take me home / to work" → navigate_saved. New destination → change_
  destination. Both PREVIEW and need the driver's yes.

PLACES — knowledgeable local, not a 5-category bot:
- find_places handles ANY place type in any language; pass `query` in the
  user's own words. Spatial intent matters: near_me («أقرب … ليا»),
  along_route («في طريقي»), near_destination («عند المكان اللي رايحينه»).
- INFORMATION (compare/recommend/show) → find_places. GOING there → add_stop.
  Reviews/hours/"is it good" for ONE place → place_details.
- Recommend like a friend: weigh rating AND review count, open-now, price,
  and the detour cost («٣ دقايق زيادة بس»).

ACTIONS — you can DO things; the tool result's "commit" field tells you how to
speak (a driving assistant minimizes confirmations — like Google Maps):
  commit="done"    → ALREADY EXECUTED. Past-tense ack + the effect. An Undo
                     pill is on screen — do NOT ask yes/no.
  commit="auto"    → executing after a short countdown. Present progressive
                     with the cost («بضيف ماستر، ٤ دقايق زيادة — قول إلغاء لو
                     مش عايز»). No question.
  commit="confirm" → PREVIEW awaiting yes. ONE short yes/no with the added
                     time. Never claim it's done.
  commit="ask"     → high-stakes (end navigation). One clear yes/no.
- Call the tool RIGHT AWAY when the request is clear. Never ask first, then
  call, then ask again.
- MULTI-INTENT («شيل الزحمة وقفلي على أقرب بنزينة»): call the tools one after
  another in the SAME turn, then speak ONE coherent summary in their order.
  At most one action can await confirmation — if a second would, do the first
  and say you'll do the other right after.
- "Cancel the stop" ≠ "end the trip": remove_stop keeps navigating;
  cancel_navigation kills the session. In doubt → remove_stop.
- Never silently fail: if a tool finds nothing or errors, say what happened
  and offer the nearest alternative. If something can't be done while
  driving, say so plainly instead of pretending.
- place_name is ONLY for a proper name the user actually said. A generic
  («بنزينة», "pharmacy") is a category — pass it as query, never place_name.
- If a result says name_match "weak", do NOT present it as their place — say
  what you found and ask ONE short question.

CONFIRMATIONS — interpret like a human:
- While an action is pending the reply can be anything: yes/no, a refinement
  («لا التانية»), a correction, or a brand-new request. Extract the real
  decision; a correction = rejection PLUS the corrected search in the SAME
  turn. Never read a full sentence as a bare "no".

MEMORY:
- The conversation history is this trip's shared memory. Refer back naturally.
- Something the user just DECLINED or cancelled: don't re-suggest it this
  trip unless they bring it up.

SAFETY:
- The driver is driving. Keep it short; never require reading; refuse
  anything that needs their eyes off the road for long, and offer the
  hands-free alternative."""

_TOOL_FOLLOWUP_V2 = (
    "Tool result(s) received. Reply in the pinned language, 1–2 short spoken "
    "sentences covering ALL executed tools in order. Match each 'commit': "
    "done → past-tense ack (Undo is on screen, no yes/no); auto → present "
    "progressive with the cost and that they can cancel; confirm → exactly "
    "ONE yes/no, never claim it's done; ask → one clear yes/no. Weave in "
    "added_min naturally when present; never guess a delta that isn't there. "
    "If a result found nothing or errored, say so plainly and offer the "
    "closest alternative."
)

# ── v2 tool schemas ───────────────────────────────────────────────────────────
_T = base._tool

_NEW_TOOLS: List[Dict] = [
    _T("traffic_check",
       "Compare LIVE congestion on the remaining route against the learned "
       "baseline for this day-of-week and hour (RouteMind's prediction model "
       "+ historical junction data). THE tool for «الزحمة دي عادية الوقت ده "
       "ولا لأ؟» / 'is this traffic normal right now?' / 'is it usually like "
       "this?'. Returns a verdict (worse/normal/lighter than usual), the "
       "junction coverage, and the live delay when known.", {}, []),
    _T("route_options",
       "Set route avoidance preferences and reroute NOW with them (executes "
       "immediately, Undo on screen). Use for 'avoid tolls', 'no highways', "
       "«من غير رسوم», «بلاش الطريق السريع». reset=true clears avoidances.",
       {"avoid_tolls": {"type": "boolean"},
        "avoid_motorways": {"type": "boolean"},
        "reset": {"type": "boolean"}}, []),
    _T("reroute_now",
       "Recalculate the route from the current position right away (fresh "
       "traffic). Use for 'recalculate', 'reroute', «حدث الطريق», «اعمل "
       "مسار من جديد».", {}, []),
    _T("avoid_jam",
       "Get around the congestion ahead IF a genuinely faster alternative "
       "exists — switches immediately with Undo when one does; otherwise "
       "reports honestly that the current route is still best and what the "
       "jam costs. Use for 'avoid this traffic', «عديني من الزحمة دي».",
       {}, []),
    _T("undo_route_change",
       "Restore the route as it was before the last copilot route change "
       "(switch/avoid/reroute). Use for 'go back to the previous route', "
       "«رجعني للطريق اللي كنت عليه».", {}, []),
    _T("navigate_saved",
       "Change the destination to the user's saved Home or Work (previews "
       "and asks for confirmation). Use for 'take me home', «وديني البيت», "
       "'to work', «على الشغل».",
       {"place": {"type": "string", "enum": ["home", "work"]}}, ["place"]),
    _T("change_destination",
       "Change the trip's destination to a new place (previews the new "
       "route, asks for confirmation). Use ONLY when they want to GO "
       "somewhere ELSE instead of the current destination — not for adding "
       "a stop on the way.",
       {"query": {"type": "string",
                  "description": "The new destination in the user's words."}},
       ["query"]),
    _T("skip_next_stop",
       "Remove the NEXT stop from the route and continue to what follows "
       "(executes immediately, Undo on screen). Use for 'skip the next "
       "stop', «عدي الوقفة الجاية».", {}, []),
    _T("save_parking",
       "Save the CURRENT location as where the user parked («احفظ مكان "
       "الركنة», 'remember where I parked').", {}, []),
    _T("where_parked",
       "Where the user's car is parked (from the saved parking memory). Use "
       "for «انا راكن فين», 'where did I park'.", {}, []),
    _T("set_voice_volume",
       "Make the guidance/assistant voice louder or quieter.",
       {"direction": {"type": "string", "enum": ["louder", "quieter"]}},
       ["direction"]),
    _T("repeat_instruction",
       "Repeat the last spoken navigation instruction («قول تاني», 'repeat "
       "that', 'what did you say').", {}, []),
    _T("set_map_theme",
       "Switch the map's day/night look.",
       {"mode": {"type": "string", "enum": ["dark", "light", "auto"]}},
       ["mode"]),
    _T("share_eta",
       "Open the share sheet with the trip's ETA and destination so the user "
       "can send it («ابعت وصولي», 'share my ETA').", {}, []),
    _T("remind_before_arrival",
       "Set a spoken reminder shortly before arrival ('remind me 5 minutes "
       "before we arrive', «فكرني قبل ما نوصل بعشر دقايق»). minutes_before=0 "
       "cancels a reminder.",
       {"minutes_before": {"type": "integer"}}, ["minutes_before"]),
    _T("call_place",
       "Look up a place's phone number and open the dialer. place_name for a "
       "named place; omit it to call the DESTINATION.",
       {"place_name": {"type": "string"}}, []),
]

# switch_route gains selector="fastest" — replace its schema in the v2 list.
def _tools_v2() -> List[Dict]:
    out = []
    for tl in base._TOOLS:
        if tl["function"]["name"] == "switch_route":
            out.append(_T(
                "switch_route",
                "Switch to an alternative route — executes immediately with "
                "an Undo. Pass index for 'the first/second one'; road_name "
                "when they name a road; selector='fastest' for 'the fastest "
                "route' / «أسرع طريق» (the server picks the best alternative "
                "or reports the current route is already fastest).",
                {"index": {"type": "integer",
                           "description": "1-based index from the alternatives"},
                 "road_name": {"type": "string"},
                 "selector": {"type": "string", "enum": ["fastest"]}}, []))
        else:
            out.append(tl)
    return out + _NEW_TOOLS


TOOLS_V2 = _tools_v2()

# Tools that can ARM a confirmation on the client — at most one armed per turn
# (the post-execution requires_confirm check is the authoritative gate; this
# set just skips executing tools whose confirm slot is already taken).
_CONFIRM_TOOLS = {"add_stop", "reroute_via", "change_destination",
                  "navigate_saved", "cancel_navigation"}


# ── Sunrise / sunset (NOAA simplified — pure math, no network) ────────────────
def _sun_times(lat: float, lng: float, utc_offset_min: int,
               now_utc: Optional[datetime] = None) -> Optional[Tuple[str, str, bool]]:
    """(sunrise 'H:MM', sunset 'H:MM', is_night) in local time; None on math
    failure (polar edge cases can't happen in Egypt but stay safe)."""
    try:
        now = now_utc or datetime.now(timezone.utc)
        local = now + timedelta(minutes=utc_offset_min)
        n = local.timetuple().tm_yday
        lat_r = math.radians(lat)
        # solar declination
        decl = math.radians(-23.44) * math.cos(2 * math.pi / 365.0 * (n + 10))
        cos_h = -math.tan(lat_r) * math.tan(decl)
        if not -1.0 <= cos_h <= 1.0:
            return None
        h = math.degrees(math.acos(cos_h)) / 15.0          # half-day length, hours
        # solar noon in local mean time ≈ 12:00 − lng/15 + offset-from-UTC
        noon = 12.0 - lng / 15.0 + utc_offset_min / 60.0
        rise, set_ = noon - h, noon + h
        def fmt(x: float) -> str:
            x %= 24.0
            hh = int(x)
            mm = int(round((x - hh) * 60)) % 60
            return f"{hh}:{mm:02d}"
        now_h = local.hour + local.minute / 60.0
        is_night = not (rise <= now_h < set_)
        return fmt(rise), fmt(set_), is_night
    except Exception:
        return None


# ── Journey context v2 formatter ──────────────────────────────────────────────
def _fmt_min_delta(m: int) -> str:
    return f"{'+' if m >= 0 else ''}{m} min"


def _format_context_v2(ctx: Dict[str, Any]) -> str:
    """Compact English trip block (English regardless of reply language —
    matches the proven pattern; the model formats numbers per the prompt)."""
    if not ctx:
        return ("[Current trip data]\nNo live trip data is available right "
                "now (navigation may not have started, or GPS was lost). Be "
                "honest about that — answer only what needs no trip data.")
    L: List[str] = ["[Current trip data]"]

    def pick(*keys):
        for k in keys:
            v = ctx.get(k)
            if v not in (None, "", []):
                return v
        return None

    def add(label: str, *keys, suffix: str = ""):
        v = pick(*keys)
        if v is not None:
            L.append(f"{label}: {v}{suffix}")

    add("Destination", "dest_name", "destination_name")
    add("Remaining distance", "remaining_km", "remaining_distance_km", suffix=" km")
    add("ETA", "eta_min", "remaining_time_min", suffix=" min")

    # Trip progress & schedule
    started = pick("trip_started_min_ago")
    covered = pick("distance_covered_km")
    if started is not None or covered is not None:
        bits = []
        if started is not None:
            bits.append(f"started {int(started)} min ago")
        if covered is not None:
            bits.append(f"{covered} km covered")
        L.append("Progress: " + ", ".join(bits))
    sched = pick("schedule_delta_min")
    if sched is not None:
        s = int(sched)
        if s >= 3:
            L.append(f"Schedule: running ~{s} min BEHIND the original ETA")
        elif s <= -3:
            L.append(f"Schedule: running ~{-s} min AHEAD of the original ETA")
        else:
            L.append("Schedule: on time vs the original ETA")

    add("Current speed", "speed_kmh", suffix=" km/h")
    lim = pick("speed_limit_kmh")
    if lim is not None:
        spd = pick("speed_kmh")
        over = ""
        try:
            if spd is not None and float(spd) > float(lim) + 3:
                over = " (user is OVER the limit)"
        except (TypeError, ValueError):
            pass
        L.append(f"Speed limit here: {lim} km/h{over}")
    add("Current road", "current_road")

    if ctx.get("next_maneuver"):
        d = ctx.get("next_maneuver_m")
        L.append("Next maneuver: " + str(ctx["next_maneuver"])
                 + (f" in {int(d)} m" if d is not None else ""))
    if ctx.get("second_maneuver"):
        L.append(f"Then: {ctx['second_maneuver']}")

    add("Traffic delay vs typical", "traffic_delay_min", suffix=" min")
    segs = ctx.get("traffic_segments") or []
    if segs:
        def seg_line(s):
            if not isinstance(s, dict):
                return str(s)
            try:
                km = base._speak_km(float(s.get("distance_km",
                                               s.get("distance_ahead_km"))))
            except (TypeError, ValueError):
                km = "?"
            road = s.get("road") or None
            try:
                length = base._speak_km(float(s.get("length_km")))
            except (TypeError, ValueError):
                length = None
            line = f"{s.get('level', '?')} in {km}"
            if road:
                line += f" on {road}"
            if length:
                line += f" (stretch ~{length})"
            return line
        L.append("Traffic ahead: " + " | ".join(seg_line(s) for s in segs[:5]))
        try:
            rank = {"moderate": 1, "heavy": 2, "severe": 3}
            worst = max((s for s in segs if isinstance(s, dict)),
                        key=lambda s: rank.get(s.get("level"), 0))
            L.append("Worst jam: " + seg_line(worst))
        except ValueError:
            pass
    elif pick("remaining_km", "remaining_distance_km") is not None:
        L.append("Traffic ahead: none detected — remaining route currently clear")

    cams = ctx.get("cameras_ahead")
    if cams is not None:
        if not cams:
            L.append("Speed cameras ahead: none on the remaining route")
        else:
            def cam_line(c):
                if not isinstance(c, dict):
                    return str(c)
                km = c.get("distance_km", c.get("distance_ahead_km", "?"))
                lm = c.get("limit", c.get("limit_kmh"))
                return (f"{c.get('type', 'camera')} in {km} km"
                        + (f", limit {lm}" if lm else ""))
            L.append(f"Speed cameras ahead: {len(cams)} total — "
                     + " | ".join(cam_line(c) for c in cams[:6]))

    br = ctx.get("bridges_ahead")
    if isinstance(br, dict) and br.get("count"):
        line = f"Bridges/flyovers ahead: {br['count']}"
        if br.get("next_km") is not None:
            line += f" — next in {br['next_km']} km"
            if br.get("next_action"):
                line += f" ({br['next_action']})"
        L.append(line)

    stops = ctx.get("stops") or []
    if stops:
        L.append("Stops already added: " + ", ".join(
            s.get("name", "?") if isinstance(s, dict) else str(s)
            for s in stops))

    alts = ctx.get("alternatives") or []
    if alts:
        def alt_line(a):
            if not isinstance(a, dict):
                return str(a)
            line = f"#{a.get('index', '?')}"
            if a.get("via"):
                line += f" via {a['via']}"
            dm = a.get("delta_min")
            if dm is not None:
                line += f" ({_fmt_min_delta(int(dm))}"
                dk = a.get("delta_km")
                if dk is not None:
                    line += f", {'+' if dk >= 0 else ''}{dk} km"
                line += " vs current)"
            if a.get("leg_to"):
                line += f" [to {a['leg_to']}]"
            return line
        L.append("Alternative routes: " + " | ".join(alt_line(a) for a in alts[:3]))

    saved = ctx.get("saved_places") or {}
    have = [k for k in ("home", "work") if isinstance(saved.get(k), dict)]
    if have:
        L.append("Saved places available: " + ", ".join(have))
    park = ctx.get("parking")
    if isinstance(park, dict) and park.get("lat") is not None:
        line = "Car parked at a saved spot"
        if park.get("age_min") is not None:
            line += f" ({int(park['age_min'])} min ago)"
        L.append(line)

    add("Route avoidances active", "avoids")
    add("Map view", "view_mode")
    if ctx.get("voice_muted") is not None:
        L.append(f"Guidance voice muted: {ctx['voice_muted']}")
    add("Local time", "local_time")

    lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
    off = ctx.get("utc_offset_min")
    if lat is not None and lng is not None and off is not None:
        sun = _sun_times(float(lat), float(lng), int(off))
        if sun:
            rise, set_, night = sun
            L.append(f"Sun: rises {rise}, sets {set_}"
                     + (" — it is night now" if night else ""))

    age = ctx.get("gps_age_s")
    try:
        if age is not None and float(age) > 15:
            L.append(f"WARNING: GPS fix is {int(float(age))} s old — position "
                     "data may be stale; say so if asked about location.")
    except (TypeError, ValueError):
        pass
    return "\n".join(L)


# ── traffic_check: live vs learned baseline ───────────────────────────────────
def _route_latlng(ctx: Dict[str, Any]) -> List[Tuple[float, float]]:
    pts = []
    for p in (ctx.get("route") or []):
        try:
            pts.append((float(p[1]), float(p[0])))     # [[lng,lat]] → (lat,lng)
        except (TypeError, ValueError, IndexError):
            continue
    return pts


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    dlat = math.radians(b[0] - a[0])
    dlng = math.radians(b[1] - a[1])
    x = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0]))
         * math.sin(dlng / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


async def _traffic_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Live junction congestion on the remaining route vs the historical
    day-of-week/hour baseline the prediction model was trained on."""
    try:
        from model.loader import ModelLoader
        from model.tier1 import get_historical_jam, jam_to_level
        from services.tomtom import get_readings_for_junctions
    except Exception as e:                                     # pragma: no cover
        logger.error(f"traffic_check imports failed: {e}")
        return {"available": False,
                "note": "The baseline comparison isn't available right now — "
                        "say so honestly; offer the live delay if known."}

    pts = _route_latlng(ctx)
    if len(pts) < 2:
        return {"available": False,
                "note": "No route geometry in hand — the comparison needs an "
                        "active trip. Say the comparison isn't available."}
    try:
        meta = ModelLoader.get_meta()
    except Exception:
        meta = None
    if meta is None:
        return {"available": False,
                "note": "Model metadata not loaded — say the comparison isn't "
                        "available right now."}

    # Junctions within 1.5 km of the remaining polyline, ordered along it.
    cands: List[Tuple[int, dict]] = []
    try:
        for _, row in meta.iterrows():
            j = (float(row["latitude"]), float(row["longitude"]))
            best_d, best_i = 1e9, -1
            for i, p in enumerate(pts):
                d = _haversine_km(j, p)
                if d < best_d:
                    best_d, best_i = d, i
            if best_d <= 1.5:
                cands.append((best_i, {
                    "junction_id": str(row["junction_id"]),
                    "latitude": j[0], "longitude": j[1]}))
    except Exception as e:
        logger.error(f"traffic_check junction scan failed: {e}")
        return {"available": False,
                "note": "Junction lookup failed — say the comparison isn't "
                        "available right now."}
    cands.sort(key=lambda c: c[0])
    # Space them out; cap the live calls at 10.
    picked: List[dict] = []
    last_i = -10
    for i, j in cands:
        if i - last_i >= max(1, len(pts) // 20):
            picked.append(j)
            last_i = i
        if len(picked) >= 10:
            break
    if not picked:
        return {"available": False, "covered_junctions": 0,
                "note": "No monitored junctions on this route — the learned "
                        "baseline doesn't cover it. Offer the live delay "
                        "from the trip data instead, if present."}

    try:
        readings = await asyncio.wait_for(
            get_readings_for_junctions(picked), timeout=6.0)
    except Exception as e:
        logger.warning(f"traffic_check live readings failed: {e}")
        readings = {}
    if not readings:
        return {"available": False, "covered_junctions": 0,
                "note": "Live junction data didn't come back — say the "
                        "comparison isn't available right now."}

    off = int(ctx.get("utc_offset_min") or 120)
    now_local = datetime.now(timezone.utc) + timedelta(minutes=off)
    lives, hists = [], []
    worst = None
    for j in picked:
        r = readings.get(j["junction_id"])
        if not r:
            continue
        live = float(r.get("jam_factor", 0.0)) / 10.0
        hist = float(get_historical_jam(j["junction_id"],
                                        now_local.replace(tzinfo=None)))
        lives.append(live)
        hists.append(hist)
        if worst is None or live > worst[1]:
            worst = (j["junction_id"], live, hist)
    if not lives:
        return {"available": False, "covered_junctions": 0,
                "note": "Live junction data didn't come back — say the "
                        "comparison isn't available right now."}

    avg_live = sum(lives) / len(lives)
    avg_hist = sum(hists) / len(hists)
    delta = avg_live - avg_hist
    if delta > 0.12:
        verdict = "worse_than_usual"
    elif delta < -0.12:
        verdict = "lighter_than_usual"
    else:
        verdict = "normal_for_this_time"
    out: Dict[str, Any] = {
        "available": True,
        "verdict": verdict,
        "covered_junctions": len(lives),
        "live_level": jam_to_level(avg_live),
        "usual_level_for_now": jam_to_level(avg_hist),
        "day_time_slot": now_local.strftime("%A %H:00 local"),
        "note": ("Answer with the verdict in plain words — e.g. 'heavier than "
                 "it usually is this hour' / «أزحم من العادي في الوقت ده» — "
                 "grounded on these measured junctions. Do not invent minute "
                 "numbers beyond what the trip data's delay field gives."),
    }
    if worst and worst[1] >= 0.4:
        out["worst_junction_live_level"] = jam_to_level(worst[1])
    dl = ctx.get("traffic_delay_min")
    if dl is not None:
        out["live_delay_min_vs_typical"] = dl
    return out


# ── New-tool executor (falls through to the legacy executor) ──────────────────
async def execute_tool_v2(name: str, args: Dict[str, Any],
                          ctx: Dict[str, Any]) -> Tuple[Dict, Optional[Dict]]:
    try:
        if name == "traffic_check":
            return await _traffic_check(ctx), None

        if name == "route_options":
            reset = bool(args.get("reset"))
            tolls = bool(args.get("avoid_tolls"))
            mways = bool(args.get("avoid_motorways"))
            if reset:
                action = {"type": "set_route_avoids", "avoid_tolls": False,
                          "avoid_motorways": False, "requires_confirm": False,
                          "undoable": True, "commit": "done"}
                return {"ok": True, "commit": "done", "reset": True,
                        "note": "Avoidances CLEARED and rerouting now (Undo on "
                                "screen). Past-tense ack."}, action
            if not tolls and not mways:
                return {"ok": False,
                        "note": "Nothing to avoid was specified — ask which "
                                "they want to avoid (tolls or highways)."}, None
            action = {"type": "set_route_avoids", "avoid_tolls": tolls,
                      "avoid_motorways": mways, "requires_confirm": False,
                      "undoable": True, "commit": "done"}
            avoiding = " and ".join([x for x, on in
                                     (("tolls", tolls), ("motorways", mways)) if on])
            return {"ok": True, "commit": "done", "avoiding": avoiding,
                    "note": "Rerouting NOW with that avoidance (Undo on "
                            "screen). Past-tense ack naming what's avoided. "
                            "If no such route exists the app keeps the "
                            "current one and says so on screen."}, action

        if name == "reroute_now":
            return {"ok": True, "commit": "done",
                    "note": "Recalculating from the current position now "
                            "(Undo on screen). Short past-tense ack."}, \
                   {"type": "reroute_now", "requires_confirm": False,
                    "undoable": True, "commit": "done"}

        if name == "avoid_jam":
            segs = ctx.get("traffic_segments") or []
            alts = ctx.get("alternatives") or []
            best = None
            for a in alts:
                if isinstance(a, dict) and a.get("delta_min") is not None \
                        and int(a["delta_min"]) < 0:
                    if best is None or int(a["delta_min"]) < int(best["delta_min"]):
                        best = a
            if best:
                action = {"type": "switch_route",
                          "index": int(best.get("index", 1)),
                          "requires_confirm": False, "undoable": True,
                          "commit": "done"}
                if best.get("via"):
                    action["via"] = best["via"]
                action["delta_min"] = int(best["delta_min"])
                return {"ok": True, "commit": "done",
                        "saves_min": -int(best["delta_min"]),
                        "via": best.get("via"),
                        "note": "SWITCHED to the faster alternative (Undo on "
                                "screen). Past-tense ack naming the minutes "
                                "saved."}, action
            delay = ctx.get("traffic_delay_min")
            return {"ok": False, "no_better_route": True,
                    "jam_exists": bool(segs),
                    **({"current_delay_min": delay} if delay is not None else {}),
                    "note": "No alternative is faster right now — the current "
                            "route is already the best. Say that honestly "
                            "(with the delay cost if known); do NOT switch."}, None

        if name == "undo_route_change":
            return {"ok": True, "commit": "done",
                    "note": "Restoring the previous route now (the app keeps "
                            "the snapshot; if none exists it says so on "
                            "screen). Past-tense ack."}, \
                   {"type": "undo_route_change", "requires_confirm": False,
                    "commit": "done"}

        if name in ("navigate_saved", "change_destination"):
            if name == "navigate_saved":
                which = args.get("place") or "home"
                saved = (ctx.get("saved_places") or {}).get(which)
                if not isinstance(saved, dict) or saved.get("lat") is None:
                    return {"found": False, "place": which,
                            "note": f"No saved {which} location exists in the "
                                    "app. Say so and suggest saving it from "
                                    "the app's saved places."}, None
                place = {"name": saved.get("name") or which.title(),
                         "lat": float(saved["lat"]), "lng": float(saved["lng"])}
            else:
                q = (args.get("query") or "").strip()
                if not q:
                    return {"found": False}, None
                hit = await base._resolve_place(ctx, q)
                if not hit:
                    return {"found": False, "requested": q,
                            "note": "Couldn't find that destination nearby — "
                                    "say so and ask for a landmark or area "
                                    "name."}, None
                place = hit
            action = {"type": "change_destination", "place": place,
                      "requires_confirm": True, "commit": "confirm"}
            return {"found": True, "new_destination": place["name"],
                    "commit": "confirm",
                    "note": "PREVIEW ONLY — the trip still heads to the old "
                            "destination until they confirm. ONE yes/no "
                            "naming the new destination."}, action

        if name == "skip_next_stop":
            stops = ctx.get("stops") or []
            if not stops:
                return {"found": False,
                        "note": "There are no stops on the route — say so."}, None
            nxt = stops[0].get("name", "?") if isinstance(stops[0], dict) \
                else str(stops[0])
            return {"ok": True, "commit": "done", "skipped": nxt,
                    "note": "Skipped — the route recalculates to what's left "
                            "(Undo on screen). Past-tense ack naming the "
                            "skipped stop."}, \
                   {"type": "remove_stop", "next": True, "name": nxt,
                    "requires_confirm": False, "undoable": True,
                    "commit": "done"}

        if name == "save_parking":
            if ctx.get("user_lat") is None:
                return {"ok": False,
                        "note": "No GPS fix right now — say the spot can't be "
                                "saved yet."}, None
            return {"ok": True, "commit": "done",
                    "note": "Parking spot SAVED at the current location (Undo "
                            "on screen). Short past-tense ack."}, \
                   {"type": "save_parking", "requires_confirm": False,
                    "undoable": True, "commit": "done"}

        if name == "where_parked":
            park = ctx.get("parking")
            if not isinstance(park, dict) or park.get("lat") is None:
                return {"found": False,
                        "note": "No parking spot is saved. Say so; they can "
                                "say 'save my parking' when parked."}, None
            res: Dict[str, Any] = {"found": True}
            if park.get("age_min") is not None:
                res["saved_min_ago"] = int(park["age_min"])
            if park.get("distance_km") is not None:
                res["distance_km"] = park["distance_km"]
            res["note"] = ("The saved spot is now shown on the map. Answer "
                           "with how long ago it was saved and the distance "
                           "when present.")
            action = {"type": "zoom_to_place",
                      "point": {"name": "Parked car",
                                "lat": float(park["lat"]),
                                "lng": float(park["lng"])},
                      "requires_confirm": False}
            return res, action

        if name == "set_voice_volume":
            d = args.get("direction") or "louder"
            return {"ok": True, "direction": d,
                    "note": "Volume adjusted — one short ack."}, \
                   {"type": "set_voice_volume", "direction": d,
                    "requires_confirm": False, "commit": "done"}

        if name == "repeat_instruction":
            return {"ok": True,
                    "note": "The app is re-speaking the last navigation "
                            "instruction right now — do NOT repeat its text "
                            "yourself; just acknowledge in 2-4 words."}, \
                   {"type": "repeat_instruction", "requires_confirm": False,
                    "commit": "done"}

        if name == "set_map_theme":
            m = args.get("mode") or "auto"
            return {"ok": True, "mode": m,
                    "note": "Theme switched — one short ack."}, \
                   {"type": "set_map_theme", "mode": m,
                    "requires_confirm": False, "commit": "done"}

        if name == "share_eta":
            eta = ctx.get("eta_min", ctx.get("remaining_time_min"))
            return {"ok": True,
                    **({"eta_min": eta} if eta is not None else {}),
                    "note": "The share sheet is opening with the ETA — tell "
                            "them to pick who to send it to."}, \
                   {"type": "share_eta", "requires_confirm": False,
                    "commit": "done"}

        if name == "remind_before_arrival":
            m = int(args.get("minutes_before") or 0)
            if m <= 0:
                return {"ok": True, "cancelled": True, "commit": "done",
                        "note": "Reminder cleared — short ack."}, \
                       {"type": "remind_before_arrival", "minutes_before": 0,
                        "requires_confirm": False, "commit": "done"}
            eta = ctx.get("eta_min", ctx.get("remaining_time_min"))
            if eta is not None and m >= int(eta):
                return {"ok": False, "eta_min": eta,
                        "note": "The trip has less time left than that — say "
                                "so and suggest a smaller number."}, None
            return {"ok": True, "minutes_before": m, "commit": "done",
                    "note": "Reminder SET — past-tense ack naming the "
                            "minutes."}, \
                   {"type": "remind_before_arrival", "minutes_before": m,
                    "requires_confirm": False, "commit": "done"}

        if name == "call_place":
            pname = (args.get("place_name") or "").strip()
            target = pname or (ctx.get("dest_name")
                               or ctx.get("destination_name") or "")
            if not target:
                return {"found": False,
                        "note": "No place to call — ask which place."}, None
            lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
            if lat is None or lng is None:
                return {"found": False,
                        "note": "No GPS fix to search near — say so."}, None
            hits = await base.search_places(lat=lat, lng=lng, query=target,
                                            limit=1)
            pid = hits[0].get("id") if hits else None
            det = await base.place_details_rich(pid) if pid else None
            phone = (det or {}).get("phone")
            if not phone:
                return {"found": False, "place": target,
                        "note": "No phone number is listed for that place — "
                                "say so honestly."}, None
            return {"found": True, "place": (det or {}).get("name", target),
                    "commit": "done",
                    "note": "The dialer is opening with the number — tell "
                            "them to tap call."}, \
                   {"type": "dial", "number": phone,
                    "place": (det or {}).get("name", target),
                    "requires_confirm": False, "commit": "done"}

        if name == "switch_route" and args.get("selector") == "fastest" \
                and args.get("index") is None and not args.get("road_name"):
            alts = ctx.get("alternatives") or []
            best = None
            for a in alts:
                if isinstance(a, dict) and a.get("delta_min") is not None:
                    if best is None or int(a["delta_min"]) < int(best["delta_min"]):
                        best = a
            if best is None:
                return {"found": False,
                        "note": "No alternatives in hand — the app can show "
                                "options via show_alternatives. Say the "
                                "current route is the only one loaded."}, None
            if int(best["delta_min"]) >= 0:
                return {"ok": False, "already_fastest": True,
                        "note": "The CURRENT route is already the fastest — "
                                "say so; do not switch."}, None
            args = {"index": int(best.get("index", 1))}
            return await base._execute_tool("switch_route", args, ctx)

        return await base._execute_tool(name, args, ctx)
    except Exception as e:
        logger.error(f"copilot v2 tool {name} failed: {e}")
        return {"error": "tool failed",
                "note": "Tell the user that didn't work and offer the nearest "
                        "alternative."}, None


# ── Deterministic translation pass (the validator's last resort) ──────────────
async def _translate(text: str, lang: str) -> Optional[str]:
    target = ("Egyptian colloquial Arabic (masri), written in Arabic script"
              if lang == "ar" else "natural English")
    messages = [
        {"role": "system",
         "content": f"Translate the in-car assistant reply below into {target}. "
                    "Keep place and brand names as they are. Keep it the same "
                    "length and tone. Output ONLY the translation."},
        {"role": "user", "content": text},
    ]
    try:
        parts: List[str] = []
        async for kind, payload in base._stream_chat(messages, with_tools=False):
            if kind == "delta":
                parts.append(payload)
        out = "".join(parts).strip()
        return out or None
    except Exception as e:
        logger.error(f"copilot translate pass failed: {e}")
        return None


# ── Language-gated generation ─────────────────────────────────────────────────
class _GateResult:
    __slots__ = ("sentences", "tool_calls", "first_bad", "full_text")

    def __init__(self):
        self.sentences: List[str] = []
        self.tool_calls: Optional[List[Dict]] = None
        self.first_bad = False          # first sentence failed the gate
        self.full_text = ""


async def _gated_pass(messages: List[Dict], lang: str, with_tools: bool,
                      emit) -> _GateResult:
    """One model pass. Streams sentences through the language gate:
      • sentences that pass are emitted immediately via emit(sentence)
      • a failing FIRST sentence aborts the pass (caller regenerates)
      • a failing LATER sentence is translated inline (or dropped if the
      translator fails — never emitted wrong)
    Returns collected state (tool calls, whether the first sentence failed)."""
    r = _GateResult()
    buf = ""
    emitted_any = False
    collected: List[str] = []
    async for kind, payload in base._stream_chat(
            messages, with_tools=with_tools,
            tools=TOOLS_V2 if with_tools else None):
        if kind == "tool_calls":
            r.tool_calls = payload
            continue
        if kind != "delta":
            continue
        buf += payload
        collected.append(payload)
        sentences, buf = split_sentences(buf, force=False)
        for s in sentences:
            if reply_lang_ok(s, lang):
                await emit(s)
                emitted_any = True
                r.sentences.append(s)
            elif not emitted_any:
                r.first_bad = True
                r.full_text = "".join(collected)
                return r
            else:
                logger.warning(f"copilot lang_fix inline: translating a "
                               f"mid-reply sentence to {lang}")
                fixed = await _translate(s, lang)
                if fixed and reply_lang_ok(fixed, lang):
                    await emit(fixed)
                    r.sentences.append(fixed)
    tail, _ = split_sentences(buf, force=True)
    for s in tail:
        if reply_lang_ok(s, lang):
            await emit(s)
            emitted_any = True
            r.sentences.append(s)
        elif not emitted_any:
            r.first_bad = True
            r.full_text = "".join(collected)
            return r
        else:
            logger.warning("copilot lang_fix inline: translating tail sentence")
            fixed = await _translate(s, lang)
            if fixed and reply_lang_ok(fixed, lang):
                await emit(fixed)
                r.sentences.append(fixed)
    r.full_text = "".join(collected)
    return r


def _lang_rule(lang: str, arabizi: bool) -> str:
    if lang == "ar":
        rule = ("RESPONSE LANGUAGE FOR THIS TURN (hard requirement): Egyptian "
                "colloquial Arabic (masri), in ARABIC SCRIPT — never فصحى, "
                "never Latin letters. Brand/place names may stay in their "
                "natural script.")
        if arabizi:
            rule += (" The user typed Arabic in Latin letters (Arabizi) — "
                     "that IS Arabic; reply in Arabic script.")
    else:
        rule = ("RESPONSE LANGUAGE FOR THIS TURN (hard requirement): natural "
                "English. Brand/place names may stay in their natural "
                "script.")
    return rule + (" This pin already honors any language request inside the "
                   "user's message — follow it exactly.")


_CORRECTIVE = ("YOUR PREVIOUS DRAFT WAS IN THE WRONG LANGUAGE AND WAS "
               "DISCARDED. Rewrite the reply ENTIRELY in the pinned language. "
               "This is a hard constraint — every sentence.")


# ── The v2 turn generator ─────────────────────────────────────────────────────
async def stream_v2(req) -> AsyncGenerator[str, None]:
    def line(obj: Dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    if not base.OPENAI_KEY:
        yield line({"t": "error", "message": "assistant not configured"})
        return

    history = [m for m in req.messages
               if m.get("role") in ("user", "assistant") and m.get("content")]
    history = history[-base.HISTORY_MAX:]
    if not history or history[-1]["role"] != "user":
        yield line({"t": "error", "message": "no user message"})
        return

    ctx = req.context or {}
    user_text = history[-1]["content"]
    res = resolve_language(user_text, prev_lang=req.prev_lang,
                           app_lang=req.app_lang)
    lang = res.lang
    t0 = time.monotonic()
    logger.info(f"copilot v2 turn: lang={lang} src={res.source} "
                f"arabizi={res.arabizi} history={len(history)}")

    # The client pins voice + UI direction off this before any text arrives.
    yield line({"t": "meta", "lang": lang, "v": 2})

    # ── Deterministic fact fast-path (no model in the loop) ──────────────────
    fp = try_fastpath(user_text, ctx, lang,
                      has_pending_action=bool(req.pending_action))
    if fp:
        logger.info("copilot v2 fastpath answer")
        yield line({"t": "delta", "text": fp})
        yield line({"t": "done", "expects_reply": False, "lang": lang})
        return

    lang_rule = _lang_rule(lang, res.arabizi)
    ctx_block = _format_context_v2(ctx)
    pending = ""
    if req.pending_action:
        pending = ("\n[Pending action awaiting user confirmation]\n"
                   + json.dumps(req.pending_action, ensure_ascii=False))

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_V2}]
    messages += history[:-1]
    messages.append({"role": "system", "content": "LANGUAGE: " + lang_rule})
    messages.append({"role": "user",
                     "content": f"{user_text}\n\n{ctx_block}{pending}"})

    out_q: asyncio.Queue = asyncio.Queue()

    async def emit(sentence: str):
        await out_q.put(sentence)

    spoke = False
    confirm_used = False
    expects_confirm = False
    emitted_text: List[str] = []

    async def run_pass(with_tools: bool) -> _GateResult:
        """gated pass + regenerate-once + translate fallback. Failing text
        never reaches the queue."""
        nonlocal spoke
        r = await _gated_pass(messages, lang, with_tools, emit)
        if r.first_bad:
            logger.warning(f"copilot lang_fix regen: first sentence not {lang}")
            messages.append({"role": "system", "content": _CORRECTIVE})
            r2 = await _gated_pass(messages, lang, with_tools, emit)
            if r2.first_bad:
                logger.warning("copilot lang_fix translate: regen failed too")
                fixed = await _translate(r2.full_text or r.full_text, lang)
                if fixed and reply_lang_ok(fixed, lang):
                    sentences, _ = split_sentences(fixed, force=True)
                    for s in sentences:
                        await emit(s)
                    r2.sentences = sentences
                    r2.first_bad = False
            return r2
        return r

    try:
        rounds = 0
        with_tools = True
        while True:
            # Run the model pass concurrently with draining the sentence queue
            # so speech streams out while the pass is still generating.
            pass_task = asyncio.create_task(run_pass(with_tools))
            while True:
                get_task = asyncio.create_task(out_q.get())
                done, _ = await asyncio.wait(
                    {pass_task, get_task},
                    return_when=asyncio.FIRST_COMPLETED)
                if get_task in done:
                    s = get_task.result()
                    spoke = True
                    emitted_text.append(s)
                    yield line({"t": "delta", "text": s + " "})
                    continue
                get_task.cancel()
                break
            result = pass_task.result()
            while not out_q.empty():
                s = out_q.get_nowait()
                spoke = True
                emitted_text.append(s)
                yield line({"t": "delta", "text": s + " "})

            if not result.tool_calls:
                break
            rounds += 1
            calls = result.tool_calls[:MAX_TOOLS_PER_ROUND]
            # One assistant message carrying ALL the round's calls, then one
            # tool message per call — the OpenAI multi-tool protocol.
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{
                                 "id": tc["id"] or f"call_{i}",
                                 "type": "function",
                                 "function": {"name": tc["name"],
                                              "arguments": tc["args"] or "{}"}}
                                 for i, tc in enumerate(calls)]})
            for i, tc in enumerate(calls):
                try:
                    args = json.loads(tc["args"] or "{}")
                except Exception:
                    args = {}
                # Spoken lead-in for the slow retrieval tools.
                if tc["name"] in ("place_details", "find_places") and not spoke:
                    lead = S("lead_details" if tc["name"] == "place_details"
                             else "lead_search", lang)
                    spoke = True
                    yield line({"t": "delta", "text": lead + " "})
                blocked_note: Dict[str, Any] = {
                    "blocked": True,
                    "note": "Another action already awaits the user's "
                            "confirmation — tell them you'll do this one "
                            "right after they decide."}
                if tc["name"] in _CONFIRM_TOOLS and confirm_used:
                    result_obj, action = blocked_note, None
                else:
                    result_obj, action = await execute_tool_v2(
                        tc["name"], args, ctx)
                    if action and action.get("requires_confirm") and confirm_used:
                        # authoritative cap: never two pending confirmations
                        result_obj, action = blocked_note, None
                if action:
                    if action.get("requires_confirm"):
                        confirm_used = True
                        expects_confirm = True
                    if action.get("commit") == "auto":
                        expects_confirm = True
                    yield line({"t": "action", "action": action})
                messages.append({"role": "tool",
                                 "tool_call_id": tc["id"] or f"call_{i}",
                                 "content": json.dumps(result_obj,
                                                       ensure_ascii=False)})
            messages.append({"role": "system",
                             "content": _TOOL_FOLLOWUP_V2 + " " + lang_rule})
            over_budget = (time.monotonic() - t0) > TURN_BUDGET_S
            with_tools = rounds < MAX_TOOL_ROUNDS and not over_budget

        if not spoke:
            yield line({"t": "delta", "text": S("fallback_ok", lang)})

        # expects_reply: a confirm/auto action is armed, or the reply asked
        # the user something (mirrors the legacy heuristic).
        full = " ".join(emitted_text).strip()
        expects = expects_confirm or full.endswith("?") or full.endswith("؟")
        yield line({"t": "done", "expects_reply": expects, "lang": lang})
    except Exception as e:
        logger.error(f"copilot v2 converse failed: {e}")
        yield line({"t": "error", "message": "upstream failure"})
