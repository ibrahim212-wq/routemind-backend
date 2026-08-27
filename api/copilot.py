"""
api/copilot.py — RouteMind Copilot v2: the streaming voice-assistant brain.

Replaces the v1 /assistant pipeline's LATENCY model wholesale:
  v1: record → upload audio → Whisper → chat (blocking) → chat#2 (blocking)
      → OpenAI TTS full synth → base64 MP3 download            (~10-15 s felt)
  v2: on-device STT (client) → THIS endpoint streams tokens the instant the
      model produces them → client speaks sentence-by-sentence with its own
      Google Cloud TTS while the rest still streams              (~1.5-3 s felt)

POST /api/copilot/converse   →  application/x-ndjson stream
(the NDJSON StreamingResponse pattern mirrors api/plan_drive_stream.py)

Request JSON:
{
  "messages":  [{"role":"user"|"assistant","content":str}, ...],  # client-held
               # trip conversation history (memory lives on the client so the
               # server stays stateless — also what makes this CarPlay/Auto
               # ready: any surface that can hold a message list can drive it)
  "context":   {...},          # rich live trip context — see _format_context()
  "app_lang":  "ar"|"en",      # app language (fallback when detection is moot)
  "pending_action": {...}|null # the action awaiting user confirmation, when the
                               # user's reply wasn't a plain yes/no (lets the
                               # model interpret "the second one" / "لا التانية")
}

Stream lines (one JSON object per line):
  {"t":"delta","text":"..."}                       # speech text, token deltas
  {"t":"action","action":{...}}                    # ≤1 per turn; client previews
  {"t":"done","expects_reply":bool,"lang":"ar"|"en"}
  {"t":"error","message":"..."}

The ACTION PROTOCOL (server resolves data, client executes):
  Tools that need server data (POI lookup, geocoding) are resolved HERE with
  the same Google/corridor machinery the Add-Stop feature uses (api/places.py),
  then emitted as an `action` whose payload the client can render directly
  (pins, preview reroute, camera). Mutating actions carry requires_confirm=true
  and the client MUST get an explicit user confirmation (voice or button)
  before executing. Camera-only actions execute immediately.

Deploy note: no new dependencies; same Cloud Run image. Redeploy the service
and the endpoint is live (OPENAI_API_KEY + GOOGLE_MAPS_API_KEY already set).
"""

import os
import json
import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.places import (resolve_nearby, _google_corridor_pois, _filter_to_corridor,
                        corridor_anchors, _mapbox_duration,
                        search_places, place_details_rich)
from services import route_geometry as geo

logger = logging.getLogger("routemind.copilot")
router = APIRouter()

# ── Config (env-overridable) ──────────────────────────────────────────────────
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
CHAT_URL       = "https://api.openai.com/v1/chat/completions"
# gpt-5.4-mini at its default reasoning effort ("none") streams the first token
# ~3x sooner than gpt-4o-mini and is far stronger at tool routing + holding
# Egyptian colloquial Arabic. Rollback is an env flip (COPILOT_MODEL=gpt-4o-mini)
# — the payload builder below adapts the params per family automatically.
COPILOT_MODEL  = os.getenv("COPILOT_MODEL", "gpt-5.4-mini")
COPILOT_TEMP   = float(os.getenv("COPILOT_TEMP", "0.8"))
MAX_TOKENS     = int(os.getenv("COPILOT_MAX_TOKENS", "180"))
HISTORY_MAX    = int(os.getenv("COPILOT_HISTORY_MAX", "24"))  # turns kept per request
POI_LIMIT      = 5                                            # per show_pois resolve

# GPT-5-family models reject `temperature` (pinned to 1) and take
# `max_completion_tokens`; style/variety lives in the prompt instead. Setting
# COPILOT_REASONING sends reasoning_effort explicitly (leave empty to use the
# model's default, which is "none" on gpt-5.4-mini — no reasoning-token delay
# before speech starts).
_GPT5_FAMILY      = COPILOT_MODEL.startswith("gpt-5")
COPILOT_REASONING = os.getenv("COPILOT_REASONING", "")

# ONE warm client for the process — connection reuse shaves 100-300 ms per turn
# vs the v1 new-AsyncClient-per-call pattern.
_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0))

# ── Personality / system prompt ───────────────────────────────────────────────
_SYSTEM = """You are the RouteMind Copilot — a sharp, warm, genuinely alive \
in-car assistant riding along on a drive in Egypt. You are NOT a narrator and \
NOT a script; you are good company who happens to be great at navigation.

VOICE & TONE (your words are spoken aloud by TTS — write for the EAR):
- 1–2 short sentences. Almost never more. The driver is driving.
- Plain spoken text only: no markdown, no lists, no emoji, no URLs.
- Match energy to the moment: upbeat when suggesting something good, calm and
  crisp for directions, quick and serious for hazards, playful for small talk.
- Vary your phrasing — never open two replies the same way in one trip.
- ANSWER THE QUESTION ASKED. Never substitute different information (traffic,
  ETA, anything) for the thing they asked about. If you can't answer, say so
  specifically or ask ONE precise clarifying question — never change topic.

SPOKEN NUMBERS — you are heard, not read. TTS mangles decimals:
- NEVER speak a decimal ("1.4", "27.8"). Round and phrase for the ear:
  1.4 km → "about a kilometer and a half" / «كيلو ونص تقريبًا»; 750 m → "about
  750 meters" / «حوالي ٧٥٠ متر»; 27.8 km → "about 28 kilometers" / «حوالي ٢٨
  كيلو»; 4 min → "about four minutes" / «حوالي ٤ دقايق».
- Prefer TIME over distance when both are known ("about ten minutes ahead").
- One rounded number per fact — never recite a list of raw figures.

LANGUAGE — you are FULLY BILINGUAL (Egyptian Arabic + English):
- Mirror the user's language PER MESSAGE: Arabic in → Egyptian colloquial
  Arabic out (masri — «عايز، دلوقتي، فاضلك، خد بالك» — NEVER فصحى/MSA). English
  in → natural English out.
- CODE-SWITCHING IS NORMAL HERE. Egyptians mix Arabic and English in one
  sentence ("add stop على أقرب gas station") — understand the mix ALWAYS, and
  reply in whichever language dominates their message. Inside an Arabic reply,
  keep English brand/place names in English — that's how Egyptians actually
  talk. Never fail, reset, or ask for clarification just because the input
  mixed languages.
- Write Arabic in Arabic script (never franco/Latin letters) — your words are
  synthesized by an Arabic voice.
- If the user asks for a specific language ("speak Arabic", "بالعربي"), switch
  to it and STAY in it until they switch again. NEVER claim you cannot speak
  Arabic or English — you speak both natively.

EARS — what you receive is a NOISY in-car voice transcript, not typed text:
- Expect mishears, dropped words, odd spellings, franco-Arabic, and English
  words rendered in Arabic script («ماستر» = Master). NEVER answer a normal
  request with "what do you mean by X" — infer the most plausible
  driving-related intent from context and act on it.
- If a key detail is GENUINELY ambiguous, ask ONE short, specific question
  that carries your best guess («تقصد ماستر بتاعة محطات البنزين؟» / "you mean
  Master, the fuel stop?") — never a generic "can you repeat that".

LOCAL KNOWLEDGE — you know Cairo & Giza like a native:
- Fuel/rest-stop brands: Master, Chillout, On The Run, Circle K, Wataniya,
  Misr Petroleum (مصر للبترول), TotalEnergies, Mobil, Emarat Misr. When the
  user names one of these, it IS a fuel/rest stop — call add_stop with that
  place_name AND category "fuel".
- Common chains: Cilantro / Costa / Starbucks / Dunkin / Beano's (cafe);
  McDonald's / KFC / Mo'men / Cook Door / Buffalo Burger (restaurant);
  El Ezaby (العزبي) / Seif (صيف) / 19011 (pharmacy).
- Roads & areas: الدائري = Ring Road, المحور = Mehwar (26th of July axis),
  الأوتوستراد = Autostrad, كوبري أكتوبر = 6th October Bridge, صلاح سالم,
  التجمع = New Cairo / Fifth Settlement, الشيخ زايد = Sheikh Zayed.
  «بنزينة / محطة بنزين» = gas station.

TRAFFIC — be a traffic insider, not a vague narrator. Traffic is spoken ONLY
when the user asks about it or a jam materially changes their trip — NEVER as
filler and NEVER as a fallback answer to an unrelated question:
- Answer with SPECIFICS from the trip data: severity + where ("heavy in about
  3 km on the Ring Road", «زحمة تقيلة بعد ٣ كيلو على الدائري»), and the total
  delay when known ("costing you about 7 minutes"). Never a bare "some
  traffic ahead" when the data has road names and distances.
- Convert distance to feel when natural: at typical speeds, congestion ~15 km
  ahead is roughly 12–15 minutes away — "you'll hit it in about a quarter
  hour".
- LIGHT PREDICTION (heuristic, be honest): Cairo rush builds roughly 7–11 am
  and 3–8 pm Sunday–Thursday (Friday is light until ~noon, then malls/outings
  pick up). Combine with Local time from the trip data: inside a building
  window say it'll likely thicken; near the end say it usually eases soon.
  ALWAYS hedge predictions with "usually / probably" («الدنيا بتبقى» /
  «غالبًا») — NEVER present a prediction as a live measurement, and never
  invent numbers for it.
- When the user asks to SEE traffic, call show_traffic — the map highlights
  the congested stretches while you describe them.

ALTERNATIVE ROUTES — a real navigator compares, recommends, and switches:
- The trip data lists alternatives with their via roads and time/distance
  deltas. Describe TRADEOFFS in plain words ("the Ring Road option is 6 km
  longer but about 4 minutes faster"), and RECOMMEND one when there's a clear
  winner. Use the Egyptian road names your driver uses.
- When the user picks by position ("the second one") pass index; when they
  pick by NAME («حولني على الدائري», "take the desert road") pass road_name —
  and if the match comes back uncertain, ask ONE short question naming the
  closest option instead of switching on a guess.

TRUTH — never fabricate, ever:
- For anything about the trip (roads, ETA, traffic, cameras, alternatives,
  stops, distances) use ONLY the [Current trip data] block. NEVER invent or
  estimate a number, road name, or condition that isn't in the data — a
  made-up "traffic is moderate" is a serious failure.
- Read the data precisely: "Traffic ahead: none detected" means the road is
  clear — say that confidently. A missing field means you genuinely don't
  have that data — say so honestly in one short sentence, and offer what you
  DO know instead (e.g. the ETA) or ask a clarifying question.
- General questions (history, food, football, anything) — answer them! Briefly,
  and when natural, tie back to the drive.

PLACES & RECOMMENDATIONS — you are a knowledgeable local, not a 5-category bot:
- find_places handles ANY place type the driver names, in any language —
  restaurants, cafes, a mosque «جامع», a mall «مول», a supermarket, a pharmacy,
  an ATM, a park, a playstation/arcade «بلايستيشن», a hangout spot «مكان نخرج
  فيه», a quiet coffee, a hidden gem. Pass `query` in the user's OWN words; the
  server resolves the type. NEVER say "the trip data has no X" — if you can
  name it, find_places can search it.
- SPATIAL INTENT — pick `where` from the phrasing (this matters a lot):
  • near_me — "near me / around here / أقرب … ليا / جنبي" (the user's CURRENT spot)
  • along_route — "on my way / along the route / في طريقي / وأنا رايح"
  • near_destination — "near where I'm going / عند المكان اللي رايحينه / at my
    destination" (around the trip's endpoint, NOT the user's current spot)
- INFORMATION vs ACTION — the key split:
  • Answering / showing / comparing / recommending → find_places. It shows pins
    and gives you ratings, review counts, price and open-now. Compare and
    recommend out loud ("El Dahan is 4.6 with 3,000 reviews and open now; the
    other's 4.8 but only 40 reviews"). Do NOT propose adding a stop unless they
    ask to GO there.
  • Actually going / stopping there → add_stop (a confirm card appears).
- COUNT — "the nearest/best one" → count 1; "a few / options / compare" → ~5;
  "all the good ones" → up to ~10. rank: best (quality) by default, closest for
  "nearest", popular for "famous/most popular", hidden_gem for "underrated /
  a hidden gem". Use min_rating for "highly rated", open_now for "open now".
- REVIEWS / hours / "is it good" about ONE place → place_details (fetches real
  reviews + hours). Paraphrase a telling review briefly; never dump raw text.
- Recommend like a friend who knows Cairo: weigh rating AND how many reviews
  (a 4.7 with 2,000 reviews is a safer bet than 4.9 with 12), factor open-now
  and price when they matter, and tie it to the drive ("it's a 3-minute detour").

ACTIONS — you can DO things, not just talk. Three response styles, driven by
the tool result's "commit" field (a driving assistant minimizes confirmations —
it acts, and offers a quick undo, exactly like Google Maps):
- Call the tool RIGHT AWAY when the request is clear ("add the nearest gas
  station") — never ask a clarifying question first, THEN call, THEN ask again.

  1) commit="done" (switch route, report incident, remove stop) — ALREADY
     EXECUTED. Speak a short PAST-TENSE acknowledgement naming the effect
     ("Switched — about 3 minutes longer", «شيلت الوقفة، كمّلنا على وجهتك»).
     Do NOT ask a yes/no — an Undo is on screen for a few seconds if they change
     their mind. This is the DEFAULT for reversible actions.

  2) commit="auto" (a CONFIDENT add-stop) — being added automatically after a
     short on-screen countdown. Say you're adding it now, present progressive,
     WITH the cost, and that they can cancel ("Adding Master, about 4 minutes
     extra — say cancel if not", «بضيف ماستر، ٤ دقايق زيادة — قول إلغاء لو مش
     عايز»). No question.

  3) commit="confirm" (an UNCERTAIN add-stop / reroute via a place) — a PREVIEW
     awaiting the driver's yes. Say in ONE short sentence what you'll do WITH
     the added time, and ask ONE yes/no ("Add Koshari Juha, adds about 4
     minutes — okay?"). Never say it's already done. Ask ONCE, not twice.

  4) commit="ask" (end navigation) — the one high-stakes action. Ask ONE clear
     yes/no to end the ENTIRE trip; do not end it in your words.

- "Cancel/remove the stop" ≠ "end the trip": remove_stop keeps navigating to
  the destination; cancel_navigation kills the whole session. When in ANY doubt
  which one they meant, remove the stop — never end a trip on an ambiguous
  phrase.

- View-only actions (show places / traffic / alternatives / overview / zoom,
  mute, 2D-3D) happen immediately — narrate what's now on screen in one line,
  then offer the logical next step.
- place_name is ONLY for a proper name or brand the user actually said
  ("Master", "Cilantro التجمع"). A generic thing ("gas station", «بنزينة»,
  "pharmacy") is a CATEGORY — never pass it as place_name.
- READ TOOL RESULTS CRITICALLY: if the result says name_match is "weak", the
  place found may NOT be what the user asked for — do NOT present it as a
  confirmed match. Say what you actually found and ask ONE short question
  whether that's what they meant.

CONFIRMATIONS — interpret like a human, never keyword-match:
- While an action is pending, the user's reply can be ANYTHING: plain yes/no,
  a refinement ("the second one", «لا التانية»), a correction («لا ده مش
  بنزينة — هات أقرب واحدة أحط فيها بنزين»), or a brand-new request. Extract
  the real decision. A correction = rejection PLUS a new search: call the tool
  again with the corrected intent in the SAME turn. NEVER read a full sentence
  as a bare "no", and NEVER end the conversation because a reply wasn't a
  clean yes/no.

MEMORY: the conversation history is this trip's shared memory — refer back to
it naturally (stops already added, things already discussed)."""

_TOOL_FOLLOWUP = (
    "Tool result received. Reply in the user's language, 1–2 short spoken "
    "sentences. Match the result's 'commit' field: commit='done' → speak a "
    "PAST-TENSE ack naming the effect and do NOT ask yes/no (an Undo is on "
    "screen); commit='confirm' → ask exactly ONE short yes/no and DO NOT claim "
    "it's done (never 'added'/'done'/'اتضافت' — it's a preview until they "
    "confirm); commit='ask' → ask one clear yes/no to proceed. If the result "
    "carries added_min, weave it in naturally ('adds about N minutes' / "
    "'هتأخرك حوالي N دقايق'); if absent, skip the delta — never guess one. If "
    "name_match is 'weak', the place may NOT be what the user asked for: say "
    "what you found (mention an other_candidates name if one looks closer) and "
    "ask whether that's what they meant. For show_traffic, describe each "
    "stretch specifically (severity, distance ahead, road name when present). "
    "If it found nothing, say so plainly and suggest the closest alternative."
)

# ── Tools (OpenAI function-calling schema) ────────────────────────────────────
_CATEGORIES = ["fuel", "restaurant", "cafe", "atm", "parking", "pharmacy"]

def _tool(name: str, desc: str, props: Dict[str, Any], required: List[str]) -> Dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

_WHERE = ["near_me", "along_route", "near_destination"]
_RANK = ["best", "closest", "popular", "hidden_gem"]

_TOOLS: List[Dict] = [
    _tool("find_places",
          "Search for and SHOW places on the map to ANSWER a question or give a "
          "recommendation — this does NOT add a stop. Use for ANY place type "
          "the user names in any language (restaurants, cafes, a mosque, a "
          "mall, a supermarket, a pharmacy, an ATM, a park, a playstation/"
          "arcade, a hangout spot, «أقرب جامع», «مكان نخرج فيه», etc.) and for "
          "recommendation/comparison asks ('best sushi near me', 'a quiet "
          "cafe', 'a hidden gem', 'which is highest rated'). Returns each "
          "place's rating, review count, price and open-now so you can compare "
          "and recommend. Pass `query` as the user's own words.",
          {"query": {"type": "string",
                     "description": "The place description in the user's words "
                     "('sushi', 'quiet cafe', 'hangout spot', 'أقرب جامع')."},
           "where": {"type": "string", "enum": _WHERE,
                     "description": "near_me = around the user's CURRENT spot; "
                     "along_route = on the remaining drive ('on my way'); "
                     "near_destination = around where they're heading. Infer "
                     "from the user's phrasing."},
           "count": {"type": "integer",
                     "description": "How many to return: 1 for 'the nearest/"
                     "best one', ~5 for options/compare, up to ~10 for 'all the "
                     "good ones'. Default 5."},
           "open_now": {"type": "boolean"},
           "min_rating": {"type": "number",
                          "description": "e.g. 4.5 for 'highly rated'."},
           "rank": {"type": "string", "enum": _RANK,
                    "description": "best = quality (rating×reviews); closest = "
                    "distance; popular = most-reviewed/famous; hidden_gem = "
                    "great rating but under-the-radar."},
           "display": {"type": "string", "enum": ["cards", "pins"],
                       "description": "cards = the user wants to BROWSE a "
                       "concrete list of a specific type and maybe pick one "
                       "('show me the 5 nearest gas stations') — tappable "
                       "cards with Add-stop buttons appear. pins = map pins "
                       "only — use for recommendations/comparisons you narrate "
                       "('suggest somewhere', 'which is best'), qualitative "
                       "asks, and single results. Default pins."}},
          ["query", "where"]),
    _tool("place_details",
          "Fetch deeper info for ONE specific place the user is asking about — "
          "its REVIEWS (what people say), opening hours, phone, editorial blurb. "
          "Use when they ask 'is it good / what do people say / reviews / when "
          "does it close / phone number' about a named place or one you just "
          "showed. Costs more, so only for a single drilled-into place. Pass "
          "place_id when the place came from an earlier find_places result "
          "(its id field) — much faster than re-searching by name.",
          {"place_name": {"type": "string"},
           "place_id": {"type": "string",
                        "description": "The id from a prior find_places result "
                        "for this place, when available."}},
          ["place_name"]),
    _tool("add_stop",
          "Propose ADDING a stop to the route (needs confirmation). Use ONLY "
          "when the user wants to actually GO somewhere / stop there — not for "
          "informational asks. Pass `query` (any place type in the user's "
          "words) OR `place_name` for a specific named place.",
          {"query": {"type": "string",
                     "description": "Place type/description to add the nearest "
                     "of ('gas station', 'صيدلية', 'ATM', 'a good cafe')."},
           "place_name": {"type": "string",
                          "description": "A specific proper name/brand the user "
                          "named ('Master', 'Cilantro التجمع')."},
           "where": {"type": "string", "enum": _WHERE,
                     "description": "near_me / along_route / near_destination — "
                     "default along_route for a stop on the way."}},
          []),
    _tool("show_traffic",
          "Highlight the congested stretches of the remaining route ON THE MAP "
          "and move the camera to them. Use whenever the user asks to SEE the "
          "traffic ('show me the traffic', «وريني الزحمة») or asks about "
          "traffic ahead while congestion exists in the trip data.", {}, []),
    _tool("show_alternatives",
          "Display the alternative routes on the map with their time deltas. "
          "Use the alternatives in trip data to recommend the best.", {}, []),
    _tool("switch_route",
          "Propose switching to an alternative route (requires confirmation). "
          "Pass index when the user says 'the first/second one'; pass "
          "road_name when they name a road («حولني على الدائري», 'take the "
          "desert road') — the server matches it to the right alternative.",
          {"index": {"type": "integer",
                     "description": "1-based index from the alternatives list"},
           "road_name": {"type": "string",
                         "description": "Road/area the user named, if they "
                         "chose by name instead of position"}},
          []),
    _tool("reroute_via",
          "Propose rerouting so the drive passes via a named road, area or "
          "place (requires confirmation).",
          {"via": {"type": "string", "description": "Road/area/place to pass through"}},
          ["via"]),
    _tool("show_overview", "Zoom the camera out to show the whole route.", {}, []),
    _tool("zoom_to_place",
          "Move the camera to look at a named place without changing the route.",
          {"place_name": {"type": "string"}}, ["place_name"]),
    _tool("set_guidance_voice", "Mute or unmute the turn-by-turn guidance voice.",
          {"muted": {"type": "boolean"}}, ["muted"]),
    _tool("set_view_mode", "Switch the map between 2D and 3D navigation view.",
          {"mode": {"type": "string", "enum": ["2d", "3d"]}}, ["mode"]),
    _tool("report_incident",
          "Report a road incident at the current location (requires confirmation).",
          {"kind": {"type": "string",
                    "enum": ["radar", "accident", "hazard", "police", "traffic", "road_closed"]}},
          ["kind"]),
    _tool("remove_stop",
          "Remove a stop from the route (reroutes to the remaining stops / the "
          "original destination — navigation CONTINUES). Use for 'cancel/remove "
          "the stop', «شيل الوقفة», 'skip the mosque stop', 'remove all stops'. "
          "NOT for ending the trip.",
          {"stop_name": {"type": "string",
                         "description": "Which stop, when the user named one or "
                         "only one exists. Omit with all=true to clear every stop."},
           "all": {"type": "boolean",
                   "description": "true = remove every stop."}},
          []),
    _tool("cancel_navigation",
          "END THE ENTIRE TRIP — stops navigating completely. ONLY when the "
          "user explicitly wants to stop navigating («خلاص وقّف الرحلة», 'end "
          "navigation', 'stop the trip'). NEVER for removing/canceling a stop "
          "— that is remove_stop (navigation continues).", {}, []),
]

# Confirmation policy (approved, mirrors Google Maps' driving-UX):
#   confirm  → a selection card awaits an explicit yes (picks a place + commits
#              a detour). requires_confirm=true on the action.
#   done     → executes IMMEDIATELY on the client with a ~6 s Undo affordance
#              (reversible, low-risk). undoable=true; requires_confirm=false.
#   ask      → the one high-stakes action (end nav): a quick explicit confirm.
_COMMIT_CONFIRM = {"add_stop", "reroute_via"}
_COMMIT_DONE    = {"switch_route", "report_incident", "remove_stop"}
_COMMIT_ASK     = {"cancel_navigation"}
# add_stop can also resolve commit="auto": confident match → a countdown card
# auto-executes in a few seconds unless cancelled (Google's add-stop pattern).
# Kept for back-compat with any reader of the old name (client now branches on
# the action's requires_confirm + undoable flags directly).
_CONFIRM_REQUIRED = _COMMIT_CONFIRM | _COMMIT_ASK

# ── Context formatting ────────────────────────────────────────────────────────
def _format_context(ctx: Dict[str, Any]) -> str:
    """Compact English trip block (English regardless of reply language —
    matches the proven v1 pattern)."""
    if not ctx:
        return ""
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
    # Key aliases accept BOTH the copilot-v2 names and the proven Android
    # buildAiTripContextJson names, so the clients reuse their existing builder.
    add("Destination", "dest_name", "destination_name")
    add("Remaining distance", "remaining_km", "remaining_distance_km", suffix=" km")
    add("ETA", "eta_min", "remaining_time_min", suffix=" min")
    add("Current speed", "speed_kmh", suffix=" km/h")
    add("Speed limit here", "speed_limit_kmh", suffix=" km/h")
    add("Current road", "current_road")
    if ctx.get("next_maneuver"):
        d = ctx.get("next_maneuver_m")
        L.append(f"Next maneuver: {ctx['next_maneuver']}"
                 + (f" in {d} m" if d else ""))
    add("Traffic delay vs free flow", "traffic_delay_min", suffix=" min")
    segs = ctx.get("traffic_segments") or []
    if segs:
        def seg_line(s):
            if not isinstance(s, dict):
                return str(s)
            # Pre-rounded for the EAR (TTS mangles decimals like "27.8").
            try:
                km = _speak_km(float(s.get("distance_km", s.get("distance_ahead_km"))))
            except (TypeError, ValueError):
                km = "?"
            road = s.get("road") or None
            try:
                length = _speak_km(float(s.get("length_km")))
            except (TypeError, ValueError):
                length = None
            line = f"{s.get('level','?')} in {km}"
            if road:
                line += f" on {road}"
            if length:
                line += f" (stretch ~{length})"
            return line
        L.append("Traffic ahead: " + " | ".join(seg_line(s) for s in segs[:5]))
    elif pick("remaining_km", "remaining_distance_km") is not None:
        # The clients only emit segments where congestion EXISTS — an empty list
        # with live route data means the road is genuinely clear. Saying so here
        # lets the model answer "how's traffic" confidently instead of guessing.
        L.append("Traffic ahead: none detected — remaining route currently clear")
    cams = ctx.get("cameras_ahead") or []
    if cams:
        def cam_line(c):
            if not isinstance(c, dict):
                return str(c)
            km = c.get("distance_km", c.get("distance_ahead_km", "?"))
            lim = c.get("limit", c.get("limit_kmh"))
            return f"{c.get('type','camera')} in {km} km" + (f", limit {lim}" if lim else "")
        L.append("Speed cameras ahead: " + " | ".join(cam_line(c) for c in cams[:4]))
    stops = ctx.get("stops") or []
    if stops:
        L.append("Stops already added: " + ", ".join(
            s.get("name", "?") if isinstance(s, dict) else str(s) for s in stops))
    alts = ctx.get("alternatives") or []
    if alts:
        def alt_line(a):
            if not isinstance(a, dict):
                return str(a)
            line = f"#{a.get('index','?')}"
            if a.get("via"):
                line += f" via {a['via']}"
            dm = a.get("delta_min")
            if dm is not None:
                line += f" ({'+' if dm >= 0 else ''}{dm} min"
                dk = a.get("delta_km")
                if dk is not None:
                    line += f", {'+' if dk >= 0 else ''}{dk} km"
                line += " vs current)"
            if a.get("desc"):
                line += f" {a['desc']}"
            return line
        L.append("Alternative routes: " + " | ".join(alt_line(a) for a in alts[:3]))
    add("Map view", "view_mode")
    if ctx.get("voice_muted") is not None:
        L.append(f"Guidance voice muted: {ctx['voice_muted']}")
    add("Local time", "local_time")
    return "\n".join(L)


def _detect_lang(text: str, app_lang: str) -> str:
    """DOMINANT script wins. The old any-Arabic-char→ar check misread normal
    Egyptian code-switching: one Arabic word inside an English sentence forced
    a full-Arabic reply (and vice-versa was impossible). Counting letters per
    script matches what the user actually spoke most of."""
    ar = sum(1 for c in text if "؀" <= c <= "ۿ")
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    if ar and en:
        return "ar" if ar >= en else "en"
    if ar:
        return "ar"
    if en:
        return "en"
    return app_lang if app_lang in ("ar", "en") else "en"


def _is_mixed(text: str) -> bool:
    """True when the message carries BOTH Arabic and Latin letters."""
    return any("؀" <= c <= "ۿ" for c in text) \
        and any(c.isascii() and c.isalpha() for c in text)

# ── Egypt POI knowledge (grounds add_stop name searches) ─────────────────────

_CAT_GTYPE = {"fuel": "gas_station", "restaurant": "restaurant", "cafe": "cafe",
              "atm": "atm", "parking": "parking", "pharmacy": "pharmacy"}


def _norm_txt(s: str) -> str:
    """Match-normalization for POI names: lowercase, unify alef/teh-marbuta/
    alef-maqsura variants, collapse whitespace. Applied to BOTH sides of every
    comparison, so table keys below are written pre-normalized."""
    s = s.lower().strip()
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"),
                 ("ى", "ي"), ("ـ", "")):
        s = s.replace(a, b)
    return " ".join(s.split())


# Known chains, aliases in BOTH scripts (users code-switch, and STT may render
# an English brand in Arabic script — «ماستر» = Master). category is the app's
# stop-category vocabulary, which also yields the Google type constraint.
_BRANDS = [
    ("fuel", ["master", "ماستر"]),
    ("fuel", ["chillout", "chill out", "تشيل اوت", "شيل اوت"]),
    ("fuel", ["on the run", "اون ذا رن", "اون ذاران"]),
    ("fuel", ["circle k", "سيركل ك"]),
    ("fuel", ["wataniya", "watanya", "الوطنيه", "وطنيه"]),
    ("fuel", ["misr petroleum", "مصر للبترول"]),
    ("fuel", ["totalenergies", "total", "توتال"]),
    ("fuel", ["mobil", "موبيل"]),
    ("fuel", ["emarat misr", "امارات مصر"]),
    ("cafe", ["cilantro", "سيلانترو"]),
    ("cafe", ["costa", "كوستا"]),
    ("cafe", ["starbucks", "ستاربكس"]),
    ("cafe", ["dunkin", "دانكن"]),
    ("cafe", ["beano", "بينوس"]),
    ("restaurant", ["mcdonald", "ماكدونالدز"]),
    ("restaurant", ["kfc", "كنتاكي"]),
    ("restaurant", ["mo'men", "momen", "مؤمن"]),
    ("restaurant", ["cook door", "كوك دور"]),
    ("restaurant", ["buffalo burger", "بافلو برجر"]),
    ("pharmacy", ["el ezaby", "ezaby", "العزبي", "عزبي"]),
    ("pharmacy", ["seif", "صيف"]),
    ("pharmacy", ["19011"]),
]


def _brand_info(text: str):
    """(category, aliases) of the first known Egypt chain mentioned, else
    (None, None)."""
    n = _norm_txt(text)
    for cat, aliases in _BRANDS:
        if any(a in n for a in aliases):
            return cat, aliases
    return None, None


# Generic category phrases the model must never free-text search — a text
# search for "gas station" matches ANY business containing "gas" (the
# gas-fitting-shop bug). Keys pre-normalized per _norm_txt.
_GENERIC_CATEGORY = {
    "gas station": "fuel", "petrol station": "fuel", "fuel station": "fuel",
    "gas": "fuel", "fuel": "fuel", "petrol": "fuel",
    "بنزينه": "fuel", "بنزين": "fuel", "محطه بنزين": "fuel",
    "محطه وقود": "fuel", "محطه": "fuel",
    "pharmacy": "pharmacy", "drugstore": "pharmacy",
    "صيدليه": "pharmacy", "اجزخانه": "pharmacy",
    "atm": "atm", "cash machine": "atm", "صراف": "atm", "صراف الي": "atm",
    "restaurant": "restaurant", "مطعم": "restaurant",
    "cafe": "cafe", "coffee shop": "cafe", "coffee": "cafe",
    "كافيه": "cafe", "قهوه": "cafe",
    "parking": "parking", "car park": "parking",
    "موقف": "parking", "باركينج": "parking", "جراج": "parking",
}


def _category_for_place_name(q: str) -> Optional[str]:
    """The category a 'place name' really is, when it's a generic phrase
    ('the nearest gas station' → fuel); None for genuine proper names."""
    n = _norm_txt(q)
    changed = True
    while changed:
        changed = False
        for lead in ("the ", "a ", "an ", "nearest ", "closest ", "اقرب "):
            if n.startswith(lead):
                n = n[len(lead):]
                changed = True
    return _GENERIC_CATEGORY.get(n)


def _name_match(query: str, name: str) -> str:
    """'strong' when the found place plausibly IS what the user named —
    substring hit, cross-script brand hit («ماستر» ↔ "Master"), or ≥half the
    query tokens present. 'weak' tells the model to ask, not assert."""
    qn, nn = _norm_txt(query), _norm_txt(name)
    if not qn or qn in nn:
        return "strong"
    _, aliases = _brand_info(qn)
    if aliases and any(a in nn for a in aliases):
        return "strong"
    qt, nt = set(qn.split()), set(nn.split())
    return "strong" if qt and len(qt & nt) / len(qt) >= 0.5 else "weak"

# ── Place-type resolution (natural language → Google place type) ──────────────
# A THIN precision layer on top of free-text Text Search: when the user names a
# clean category (in Arabic/English/colloquial) we constrain the search to the
# matching Google type; anything else ("hangout spot", "playstation", "somewhere
# quiet") flows through as a free-text semantic query (Text Search handles those
# well). NOT an exhaustive enum — Google has ~300 types; this maps the driver's
# common needs + Egyptian colloquial terms, and everything else stays free-text.
_TYPE_VOCAB = {
    # fuel
    "gas station": "gas_station", "petrol station": "gas_station", "gas": "gas_station",
    "fuel": "gas_station", "بنزينه": "gas_station", "محطه بنزين": "gas_station",
    "محطه وقود": "gas_station", "بنزين": "gas_station",
    # food & drink
    "restaurant": "restaurant", "restaurants": "restaurant", "مطعم": "restaurant",
    "مطاعم": "restaurant", "food": "restaurant", "اكل": "restaurant",
    "cafe": "cafe", "coffee": "cafe", "coffee shop": "cafe", "كافيه": "cafe",
    "كافيهات": "cafe", "قهوه": "cafe", "café": "cafe",
    "fast food": "meal_takeaway", "takeaway": "meal_takeaway", "تيك اواي": "meal_takeaway",
    "bar": "bar", "بار": "bar", "bakery": "bakery", "مخبز": "bakery", "فرن": "bakery",
    "ice cream": "ice_cream_shop", "ايس كريم": "ice_cream_shop", "جيلاتي": "ice_cream_shop",
    # worship
    "mosque": "mosque", "جامع": "mosque", "مسجد": "mosque", "جوامع": "mosque", "مساجد": "mosque",
    "church": "church", "كنيسه": "church", "كنيسة": "church",
    # shopping
    "mall": "shopping_mall", "مول": "shopping_mall", "مولات": "shopping_mall",
    "shopping mall": "shopping_mall", "سنتر": "shopping_mall",
    "supermarket": "supermarket", "سوبر ماركت": "supermarket", "سوبرماركت": "supermarket",
    "hypermarket": "supermarket", "هايبر": "supermarket",
    "grocery": "grocery_store", "بقاله": "grocery_store", "بقاله صغيره": "convenience_store",
    "convenience store": "convenience_store", "supermarket small": "convenience_store",
    "electronics": "electronics_store", "الكترونيات": "electronics_store",
    "phone shop": "electronics_store", "محل موبايلات": "electronics_store",
    "clothes": "clothing_store", "ملابس": "clothing_store",
    # health & finance
    "pharmacy": "pharmacy", "صيدليه": "pharmacy", "اجزخانه": "pharmacy", "drugstore": "pharmacy",
    "hospital": "hospital", "مستشفى": "hospital", "مستشفي": "hospital",
    "clinic": "doctor", "عياده": "doctor", "دكتور": "doctor",
    "atm": "atm", "cash machine": "atm", "صراف": "atm", "صراف الي": "atm", "ماكينه صراف": "atm",
    "bank": "bank", "بنك": "bank",
    # automotive
    "parking": "parking", "car park": "parking", "موقف": "parking",
    "باركينج": "parking", "جراج": "parking",
    "car wash": "car_wash", "مغسله": "car_wash", "غسيل سيارات": "car_wash",
    "car repair": "car_repair", "ميكانيكي": "car_repair", "ورشه": "car_repair",
    # recreation / hangout
    "park": "park", "حديقه": "park", "جنينه": "park", "حديقة": "park",
    "cinema": "movie_theater", "سينما": "movie_theater", "movie theater": "movie_theater",
    "gym": "gym", "جيم": "gym", "نادي": "gym",
    "arcade": "video_arcade", "playstation": "video_arcade", "بلايستيشن": "video_arcade",
    "بلاي ستيشن": "video_arcade", "بلاي": "video_arcade",
    "bowling": "bowling_alley", "بولينج": "bowling_alley",
    "amusement": "amusement_center", "ملاهي": "amusement_center", "playground": "amusement_center",
    "zoo": "zoo", "حديقه حيوان": "zoo",
    "museum": "museum", "متحف": "museum",
    "hotel": "hotel", "فندق": "hotel",
    # transport
    "train station": "train_station", "محطه قطر": "train_station",
    "bus station": "bus_station", "موقف اتوبيس": "bus_station",
}
# Broad umbrella queries with NO single clean type → let the model's free text
# drive Text Search; we just flag them so we don't wrongly type-constrain.
_OPEN_ENDED = {"hangout", "hang out", "نخرج", "نقعد", "قعده", "مكان حلو",
               "somewhere", "place to", "fun", "entertainment", "ترفيه", "تسليه"}


def _resolve_place_type(query: str) -> Optional[str]:
    """Google place type to CONSTRAIN a Text Search, or None to leave it fully
    free-text/semantic. Checks a known brand's category, then the vocabulary
    (whole string, then leading noun after stripping 'nearest/aقرب' etc.)."""
    n = _norm_txt(query)
    if any(w in n for w in _OPEN_ENDED):
        return None
    bcat, _al = _brand_info(n)
    if bcat:
        return _CAT_GTYPE.get(bcat)
    if n in _TYPE_VOCAB:
        return _TYPE_VOCAB[n]
    changed = True
    while changed:
        changed = False
        for lead in ("the ", "a ", "an ", "nearest ", "closest ", "best ", "good ",
                     "some ", "any ", "اقرب ", "احسن ", "افضل ", "اي "):
            if n.startswith(lead):
                n = n[len(lead):]; changed = True
    if n in _TYPE_VOCAB:
        return _TYPE_VOCAB[n]
    # last: any known vocab term appearing as a token
    toks = set(n.split())
    for term, gtype in _TYPE_VOCAB.items():
        if " " not in term and term in toks:
            return gtype
    return None


# ── Rating relevance (defect: it spoke a mosque's star rating) ───────────────
# Principled rule: a rating helps when the user is CHOOSING BY QUALITY among
# comparable options (food/drink, retail, entertainment, lodging, personal
# services). It is meaningless for utility/worship/civic stops where the choice
# is proximity-driven. Prefix-based on Google primaryType, not a name list.
_RATING_TYPE_PREFIXES = (
    "restaurant", "cafe", "coffee", "bar", "bakery", "meal_", "ice_cream",
    "food", "shopping", "store", "market", "mall", "movie", "video_arcade",
    "bowling", "amusement", "karaoke", "night_club", "casino", "zoo",
    "aquarium", "museum", "tourist_attraction", "hotel", "lodging", "gym",
    "fitness", "spa", "beauty", "hair",
)


def _rating_relevant(primary_type: Optional[str]) -> bool:
    if not primary_type:
        return True   # unknown type (free-text hit) — let the data through
    t = primary_type.lower()
    return any(t.startswith(p) or p.rstrip("_") in t for p in _RATING_TYPE_PREFIXES)


# ── Speech-friendly numbers (TTS reads decimals horribly) ─────────────────────
def _speak_km(km: Optional[float]) -> Optional[str]:
    """Round a distance the way a co-pilot would SAY it: <1 km → metres to the
    nearest 50; <10 km → nearest half; ≥10 km → whole km. Returns a string the
    model can read aloud verbatim ('750 m', '1.5 km', '28 km')."""
    if km is None:
        return None
    if km < 0.975:
        m = max(50, int(round(km * 1000 / 50.0)) * 50)
        return f"{m} m"
    if km < 9.75:
        halves = round(km * 2) / 2
        return f"{int(halves)} km" if halves == int(halves) else f"{halves:.1f} km"
    return f"{int(round(km))} km"

# ── Recommendation ranking (rating × review-count, no popularity API) ─────────
# Bayesian prior: m = "phantom reviews" at the 4.0 mean. m=30 (approved
# "balanced") shrinks a 5.0/3-reviews place to ~4.1 (so it can't top a 4.6/2000
# at ~4.59), yet leaves a genuine 4.4/120 gem at ~4.32 — above the 4.3 gem gate.
_SHRINK_M = 30.0
_SHRINK_C = 4.0      # prior mean rating


def _shrunk_rating(rating: Optional[float], reviews: Optional[int]) -> float:
    """Bayesian-shrunk rating: pulls low-count ratings toward the 4.0 prior so a
    '5.0 with 3 reviews' can't beat a '4.6 with 2,000'. Unknown → neutral prior."""
    if rating is None or not reviews:
        return _SHRINK_C
    v = float(reviews)
    return (v / (v + _SHRINK_M)) * float(rating) + (_SHRINK_M / (v + _SHRINK_M)) * _SHRINK_C


def _rank_places(places: List[dict], rank: str) -> List[dict]:
    """Order enriched places per the requested lens (approved 'balanced' bands):
      closest    → straight-line/detour distance
      popular    → most reviews (famous)
      hidden_gem → shrunk-rating ≥ 4.3 AND 15 ≤ reviews ≤ 300, best first
      best       → shrunk-rating, with a small review floor so 1-review noise
                   doesn't win."""
    for p in places:
        p["_shrunk"] = round(_shrunk_rating(p.get("rating"), p.get("reviews")), 3)
    if rank == "closest":
        return sorted(places, key=lambda p: p.get("distance_m") or p.get("detour_min", 1e9) or 1e9)
    if rank == "popular":
        return sorted(places, key=lambda p: -(p.get("reviews") or 0))
    if rank == "hidden_gem":
        gems = [p for p in places if p["_shrunk"] >= 4.3
                and 15 <= (p.get("reviews") or 0) <= 300]
        return sorted(gems or places, key=lambda p: -p["_shrunk"])
    # best (default): quality first, require a minimal confidence of ≥5 reviews
    ranked = sorted(places, key=lambda p: -p["_shrunk"])
    solid = [p for p in ranked if (p.get("reviews") or 0) >= 5]
    return solid or ranked

# ── Server-side action resolvers (reuse the Add-Stop machinery) ───────────────
def _route_pts(ctx: Dict[str, Any]):
    """Context route [[lng,lat],...] → [(lat,lng),...]; [] when absent."""
    return geo.geojson_to_latlng(ctx.get("route") or [])


def _dest_point(ctx: Dict[str, Any]):
    """Destination (lat,lng): explicit dest_lat/lng if present, else the last
    point of the route geometry the client sent. None when unknown."""
    dlat, dlng = ctx.get("dest_lat"), ctx.get("dest_lng")
    if dlat is not None and dlng is not None:
        return (float(dlat), float(dlng))
    pts = _route_pts(ctx)
    return pts[-1] if pts else None


async def _annotate_detours(ctx: Dict[str, Any], places: List[dict],
                            cap: int = 6) -> None:
    """Attach detour_min (extra driving minutes to visit) to the first `cap`
    along-route candidates, concurrently. Bounded so latency/cost stay sane;
    the rest keep whatever distance ordering search returned."""
    lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
    dest = _dest_point(ctx)
    if lat is None or lng is None or dest is None:
        return
    subset = places[:cap]

    async def _one(p):
        p["detour_min"] = await _detour_added_min(ctx, p["lat"], p["lng"])
    try:
        async with httpx.AsyncClient(timeout=2.0):
            await asyncio.wait_for(asyncio.gather(*[_one(p) for p in subset]), timeout=2.5)
    except Exception:
        pass


async def _find_places(ctx: Dict[str, Any], query: str, where: str = "near_me",
                       count: int = 5, open_now: bool = False,
                       min_rating: Optional[float] = None,
                       rank: str = "best") -> List[dict]:
    """The copilot's unified place-retrieval brain. Resolves an optional Google
    type from the natural-language query, picks the bias point / route corridor
    from the spatial intent, runs ONE enriched Text Search, annotates detour
    minutes for along-route, and ranks per the requested lens."""
    q = (query or "").strip()
    if not q:
        return []
    lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
    if lat is None or lng is None:
        return []
    included = _resolve_place_type(q)
    # Google Text Search ranks along-route/relevance itself; use DISTANCE only
    # for an explicit "closest" ask so the corridor/quality order is preserved.
    api_rank = "distance" if rank == "closest" else "relevance"
    fetch = max(count, 10)          # over-fetch so ranking has candidates to sort
    poly = None
    bias = (lat, lng)
    radius = 6000.0
    if where == "along_route":
        pts = _route_pts(ctx)
        if len(pts) >= 2:
            remaining = geo.remaining_route(pts, (lat, lng))
            if len(remaining) >= 2:
                poly = geo.encode_polyline5(remaining)
        radius = 12000.0
    elif where == "near_destination":
        dest = _dest_point(ctx)
        if dest:
            bias = dest
        radius = 6000.0
    places = await search_places(
        lat=bias[0], lng=bias[1], query=q, included_type=included,
        open_now=open_now, min_rating=min_rating, rank=api_rank,
        radius_m=radius, route_polyline=poly, limit=fetch)
    if not places:
        return []
    if where == "along_route":
        await _annotate_detours(ctx, places)
        # generous ~10-min corridor (approved): keep detours ≤ 12 min when we
        # measured one; unmeasured (beyond the cap) pass through.
        keep = [p for p in places if p.get("detour_min") is None or p["detour_min"] <= 12]
        places = keep or places
    ranked = _rank_places(places, rank)
    return ranked[:count]


async def _resolve_pois(ctx: Dict[str, Any], category: str,
                        limit: int = POI_LIMIT, nearest: bool = False) -> List[Dict]:
    """Category POIs. nearest=False → along the remaining route (corridor search,
    same engine as /along-route/pois). nearest=True → closest to the user's
    current location regardless of route (circle search via resolve_nearby)."""
    gtype = _CAT_GTYPE.get(category)
    if gtype is None:
        return []
    lat = ctx.get("user_lat"); lng = ctx.get("user_lng")
    pts = _route_pts(ctx)
    try:
        if not nearest and len(pts) >= 2:
            remaining = geo.remaining_route(pts, (lat, lng)) if lat and lng else pts
            if len(remaining) >= 2:
                cum = geo.cumulative_distances(remaining)
                anchors = corridor_anchors(remaining, cum)  # gapless near + sparse far
                raw = await _google_corridor_pois(gtype, anchors)
                pois = _filter_to_corridor(raw, remaining, cum, category, limit)
                if pois:
                    return [p.dict() for p in pois]
    except Exception as e:
        logger.warning(f"copilot corridor resolve failed: {e}")
    # Nearest-to-user (explicit nearest=True, or corridor fallback): straight-line
    # closest around the user. resolve_nearby's category set is
    # {gas_station,pharmacy,atm,restaurant,cafe}; map fuel→gas_station and use a
    # text query for anything it doesn't support (e.g. parking).
    if lat is None or lng is None:
        return []
    supported = {"restaurant", "cafe", "atm", "pharmacy"}
    if category == "fuel":
        results = await resolve_nearby(lat, lng, category="gas_station", limit=limit)
    elif category in supported:
        results = await resolve_nearby(lat, lng, category=category, limit=limit)
    else:  # parking and any other → free-text search
        results = await resolve_nearby(lat, lng, query=category, limit=limit)
    return [{"id": f"near/{i}", "name": r.name, "lat": r.lat, "lng": r.lng,
             "category": category, "distance_from_route_m": 0,
             "along_route_distance_m": r.distance_m, "address": r.address}
            for i, r in enumerate(results)]


async def _resolve_places(ctx: Dict[str, Any], query: str,
                          included_type: Optional[str] = None,
                          limit: int = 3) -> List[Dict]:
    """Free-text place resolve near the user (Google text search). When
    included_type is given the search is CONSTRAINED to that Google place type
    — 'Master' with gas_station can only match fuel stations, so a random shop
    whose name merely contains the word can't win."""
    lat = ctx.get("user_lat"); lng = ctx.get("user_lng")
    if lat is None or lng is None or not query.strip():
        return []
    results = await resolve_nearby(lat, lng, query=query.strip(),
                                   included_type=included_type, limit=limit)
    return [{"name": r.name, "lat": r.lat, "lng": r.lng,
             "address": r.address, "distance_m": r.distance_m}
            for r in results]


async def _resolve_place(ctx: Dict[str, Any], query: str) -> Optional[Dict]:
    """Single best free-text hit (reroute_via / zoom_to_place)."""
    places = await _resolve_places(ctx, query, limit=1)
    return places[0] if places else None


async def _detour_added_min(ctx: Dict[str, Any], stop_lat: float,
                            stop_lng: float) -> Optional[int]:
    """Real detour cost of visiting a stop: (current→stop→dest) minus
    (current→dest), both live driving-traffic durations. Hard 1.6 s cap so the
    spoken confirm never stalls — on timeout/failure the assistant simply
    speaks without the delta (the client still shows its own +X min chip).
    Destination = last point of the remaining-route geometry the client sent."""
    lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
    route = ctx.get("route") or []
    if lat is None or lng is None or len(route) < 2:
        return None
    try:
        dlng, dlat = float(route[-1][0]), float(route[-1][1])
    except Exception:
        return None
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            base_t = _mapbox_duration(f"{lng},{lat};{dlng},{dlat}", client)
            via_t  = _mapbox_duration(f"{lng},{lat};{stop_lng},{stop_lat};{dlng},{dlat}", client)
            base, via = await asyncio.wait_for(
                asyncio.gather(base_t, via_t), timeout=1.6)
        if base is None or via is None:
            return None
        return max(0, round((via - base) / 60))
    except Exception:
        return None


_ROAD_STOPWORDS = {"road", "street", "way", "highway", "rd", "st", "the", "el",
                   "al", "one", "طريق", "شارع", "ال"}


def _match_alternative(road_name: str, alts: List[Any]) -> Optional[int]:
    """1-based index of the alternative whose 'via' roads match road_name
    (normalized token overlap, same _norm_txt rules as POI matching); None
    when no single confident match — the model must then ask, not guess.
    Generic words ("road", «طريق») don't count as evidence."""
    q = set(_norm_txt(road_name).split()) - _ROAD_STOPWORDS
    if not q:
        return None
    hits = []
    for a in alts:
        if not isinstance(a, dict):
            continue
        via = _norm_txt(str(a.get("via") or ""))
        if via and any(tok in via for tok in q):
            hits.append(int(a.get("index", 0)))
    return hits[0] if len(hits) == 1 else None


async def _execute_tool(name: str, args: Dict[str, Any],
                        ctx: Dict[str, Any]) -> (Dict, Optional[Dict]):
    """Run a tool. Returns (tool_result_for_model, action_for_client|None)."""
    try:
        if name == "find_places":
            query = (args.get("query") or "").strip()
            where = args.get("where") or "near_me"
            count = int(args.get("count") or 5)
            count = max(1, min(10, count))
            rank = args.get("rank") or ("closest" if count == 1 else "best")
            places = await _find_places(
                ctx, query, where=where, count=count,
                open_now=bool(args.get("open_now")),
                min_rating=args.get("min_rating"), rank=rank)
            if not places:
                return {"found": False, "query": query, "where": where,
                        "note": "Nothing matched nearby. Say so specifically and "
                                "offer the closest alternative (widen the area, "
                                "or the nearest of a related type)."}, None
            # Compact, model-facing brief. Ratings included ONLY where a rating
            # helps choose (food/retail/entertainment) — never for mosques,
            # ATMs and other utility stops (they'd get recited pointlessly).
            def brief(p):
                b = {"name": p["name"]}
                if p.get("id"):
                    b["id"] = p["id"]   # lets place_details skip the re-search
                if p.get("rating") is not None and _rating_relevant(p.get("primary_type")):
                    b["rating"] = p["rating"]; b["reviews"] = p.get("reviews")
                if p.get("price"):    b["price"] = p["price"]
                if p.get("open_now") is not None: b["open_now"] = p["open_now"]
                if p.get("detour_min") is not None:
                    b["detour_min"] = p["detour_min"]
                elif p.get("distance_m") is not None:
                    b["away"] = _speak_km(p["distance_m"] / 1000.0)
                return b
            display = "cards" if args.get("display") == "cards" and len(places) >= 2 \
                      else "pins"
            action = {"type": "show_places", "query": query, "where": where,
                      "places": places, "display": display,
                      "requires_confirm": False}
            return {"found": True, "where": where, "count": len(places),
                    "places": [brief(p) for p in places],
                    "note": "Shown on the map"
                            + (" as browsable cards the user can tap to add"
                               if display == "cards" else "")
                            + ". Answer/compare with what's given; recommend one "
                            "when a clear winner exists. Ratings are included "
                            "only where they matter — if absent, do NOT mention "
                            "ratings at all. This is INFORMATION — do not ask "
                            "to add a stop unless the user asks to go there."}, action

        if name == "place_details":
            pname = (args.get("place_name") or "").strip()
            pid = (args.get("place_id") or "").strip()
            if not pname and not pid:
                return {"found": False}, None
            # FAST PATH: an id from a prior find_places result skips the whole
            # re-search round-trip (the single biggest reviews-latency cost).
            if not pid:
                lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
                hit = await search_places(lat=lat, lng=lng, query=pname, limit=1) \
                    if lat is not None and lng is not None else []
                if not hit or not hit[0].get("id"):
                    return {"found": False, "requested": pname,
                            "note": "Couldn't find that place to look up."}, None
                pid = hit[0]["id"]
            det = await place_details_rich(pid)
            if not det:
                return {"found": False, "requested": pname}, None
            action = {"type": "show_places", "places": [det],
                      "query": pname, "requires_confirm": False}
            return {"found": True, "place": det.get("name"),
                    "rating": det.get("rating"), "reviews": det.get("reviews"),
                    "price": det.get("price"), "open_now": det.get("open_now"),
                    "hours": det.get("hours"), "editorial": det.get("editorial"),
                    "review_samples": det.get("review_samples"),
                    "note": "Summarize helpfully; if a review captures it well, "
                            "paraphrase ONE briefly. Don't read raw star counts "
                            "robotically."}, action

        if name == "add_stop":
            cat = None
            qname = (args.get("place_name") or "").strip()
            query = (args.get("query") or "").strip()
            where = args.get("where") or "along_route"
            nearest = where == "near_me"
            # Guardrail 1 — a generic phrase passed as place_name ("gas
            # station", «بنزينة») is really a category query, not a proper name.
            if qname and _category_for_place_name(qname):
                query, qname = qname, ""
            place = None
            match = "strong"
            others: List[str] = []
            if qname:
                # Guardrail 2 — a known Egypt chain constrains the text search
                # to its Google type (Master → gas_station) + cross-script
                # aliases for the name-match signal.
                bcat, _aliases = _brand_info(qname)
                gtype = _CAT_GTYPE.get(bcat or "")
                places = await _resolve_places(ctx, qname, included_type=gtype)
                if not places and gtype:
                    places = await _resolve_places(ctx, qname)  # relax the type
                if places:
                    place = places[0]
                    match = _name_match(qname, place["name"])
                    others = [p["name"] for p in places[1:]]
            elif query:
                # Nearest matching place of the described type (any type now,
                # not just the old 6 categories), via the unified retrieval.
                found = await _find_places(ctx, query, where=where, count=1,
                                           rank="closest")
                if found:
                    p = found[0]
                    place = {"name": p["name"], "lat": p["lat"], "lng": p["lng"],
                             "distance_m": p.get("distance_m", 0)}
            if not place:
                miss: Dict[str, Any] = {"found": False}
                if qname or query:
                    miss["requested"] = qname or query
                    miss["note"] = ("That wasn't found nearby. Offer the nearest "
                                    "place of a related type instead of "
                                    "inventing a match.")
                return miss, None
            added = await _detour_added_min(ctx, place["lat"], place["lng"])
            # CONFIDENCE POLICY (approved direction — Google auto-adds a clearly
            # requested stop after a short cancellable countdown):
            #   AUTO  = explicit+specific request (proper name or clean type),
            #           strong name match, and a KNOWN, small detour (≤8 min).
            #   CONFIRM = weak match, unknown detour cost, or a big detour.
            confident = (match == "strong" and added is not None and added <= 8
                         and bool(qname or _resolve_place_type(query or "")))
            commit = "auto" if confident else "confirm"
            action = {"type": "add_stop", "place": place,
                      "requires_confirm": not confident, "commit": commit}
            if confident:
                action["auto_seconds"] = 6
            dist_key = "km_ahead_on_route" if (query and not nearest) else "km_from_you"
            result: Dict[str, Any] = {
                "found": True, "place": place["name"], "commit": commit,
                dist_key: round(place.get("distance_m", 0) / 1000.0, 1),
                "name_match": match,
                "note": ("ADDING IT NOW (countdown on screen, cancellable). Say "
                         "you're adding it — present tense, WITH the added time "
                         "— and that they can say cancel. Do NOT ask yes/no."
                         if confident else
                         "PREVIEW ONLY — nothing added yet. Ask ONE yes/no "
                         "question to confirm; never say it's done."),
            }
            if added is not None:
                result["added_min"] = added
                action["added_min"] = added
            if match == "weak":
                result["other_candidates"] = others
                result["note"] = (
                    "CAUTION: the found place's name does not clearly match "
                    "what the user asked for. Previewed on the map, but do NOT "
                    "present it as their requested place — say what you found "
                    "and ask ONE short question whether that's what they meant.")
            return result, action

        if name == "remove_stop":
            stops = ctx.get("stops") or []
            if not stops:
                return {"found": False,
                        "note": "No stops on the route. Say so; if they meant "
                                "ending the trip they must say that explicitly."}, None
            names = [s.get("name", "?") if isinstance(s, dict) else str(s)
                     for s in stops]
            want_all = bool(args.get("all"))
            target = (args.get("stop_name") or "").strip()
            resolved = None
            if not want_all:
                if target:
                    tn = _norm_txt(target)
                    hits = [n for n in names if tn in _norm_txt(n)
                            or _norm_txt(n) in tn]
                    if len(hits) == 1:
                        resolved = hits[0]
                    elif not hits and len(names) == 1:
                        resolved = names[0]
                    elif len(hits) > 1:
                        return {"found": False, "stops": names,
                                "note": "Multiple stops match — ask which one."}, None
                elif len(names) == 1:
                    resolved = names[0]
                else:
                    return {"found": False, "stops": names,
                            "note": "Several stops exist — ask which to remove."}, None
                if resolved is None:
                    return {"found": False, "requested": target, "stops": names,
                            "note": "No stop by that name — tell them what the "
                                    "stops are and ask."}, None
            action = {"type": "remove_stop",
                      **({"all": True} if want_all else {"name": resolved}),
                      "requires_confirm": False, "undoable": True, "commit": "done"}
            return {"ok": True, "commit": "done",
                    "removed": "all stops" if want_all else resolved,
                    "note": "REMOVED — the route is recalculating to what's "
                            "left (Undo on screen). Past-tense ack; navigation "
                            "CONTINUES to the destination. Do NOT ask yes/no."}, action

        if name == "show_traffic":
            segs = ctx.get("traffic_segments") or []
            action = {"type": "show_traffic", "requires_confirm": False}
            if not segs:
                # Nothing to frame — the client shows nothing; the model says
                # the road is clear (context corroborates via 'none detected').
                return {"found": False,
                        "note": "No congestion on the remaining route — say "
                                "it's clear confidently, no map action."}, None
            return {"found": True, "segments": segs,
                    "note": "Congested stretches are now highlighted on the "
                            "map and the camera is framing them. Describe them "
                            "specifically (road name, distance ahead, "
                            "severity)."}, action

        if name == "show_alternatives":
            alts = ctx.get("alternatives") or []
            action = {"type": "show_alternatives", "requires_confirm": False}
            if not alts:
                # The client fetches fresh alternatives when it receives this
                # action with none in hand — don't tell the user "none exist".
                return {"found": False,
                        "note": "None in hand YET — the app is fetching fresh "
                                "alternatives right now. Say you're pulling up "
                                "the options and they'll appear on screen in a "
                                "moment; they can ask again to compare."}, action
            return {"found": True, "alternatives": alts,
                    "note": "Alternatives now displayed. Compare them by their "
                            "via roads and time/distance deltas; recommend one "
                            "when a clear winner exists."}, action

        if name == "switch_route":
            alts = ctx.get("alternatives") or []
            road = (args.get("road_name") or "").strip()
            idx = args.get("index")
            if idx is None and road:
                idx = _match_alternative(road, alts)
                if idx is None:
                    return {"found": False, "requested_road": road,
                            "alternatives": alts,
                            "note": "Could not confidently match that road to "
                                    "one alternative. Ask ONE short question "
                                    "naming the closest option(s) — do NOT "
                                    "switch on a guess."}, None
            idx = int(idx or 1)
            # IMMEDIATE + UNDO (reversible): the client switches at once and
            # shows a ~6 s Undo. delta_min lets the ack name the cost.
            via = next((a.get("via") for a in alts
                        if isinstance(a, dict) and a.get("index") == idx), None)
            dmin = next((a.get("delta_min") for a in alts
                         if isinstance(a, dict) and a.get("index") == idx), None)
            action = {"type": "switch_route", "index": idx,
                      "requires_confirm": False, "undoable": True, "commit": "done"}
            if via:
                action["via"] = via
            if dmin is not None:
                action["delta_min"] = dmin
            res = {"ok": True, "index": idx, "commit": "done",
                   **({"via": via} if via else {}),
                   "note": "SWITCHED NOW (Undo is on screen). Speak a short "
                           "past-tense ack; if delta_min is present name the "
                           "time change. Do NOT ask yes/no."}
            if dmin is not None:
                res["delta_min"] = dmin
            return res, action

        if name == "reroute_via":
            via = args.get("via", "")
            place = await _resolve_place(ctx, via)
            if not place:
                return {"found": False, "via": via}, None
            action = {"type": "reroute_via", "point": place,
                      "requires_confirm": True, "commit": "confirm"}
            result = {"found": True, "via": place["name"], "commit": "confirm",
                      "note": "PREVIEW ONLY — not rerouted yet. Ask ONE yes/no "
                              "question; never say it's done."}
            added = await _detour_added_min(ctx, place["lat"], place["lng"])
            if added is not None:
                result["added_min"] = added
                action["added_min"] = added
            return result, action

        if name == "zoom_to_place":
            place = await _resolve_place(ctx, args.get("place_name", ""))
            if not place:
                return {"found": False}, None
            action = {"type": "zoom_to_place", "point": place,
                      "requires_confirm": False}
            return {"found": True, "place": place["name"]}, action

        if name == "show_overview":
            return {"ok": True}, {"type": "show_overview", "requires_confirm": False}
        if name == "set_guidance_voice":
            m = bool(args.get("muted", True))
            return {"ok": True, "muted": m}, {"type": "set_guidance_voice",
                                              "muted": m, "requires_confirm": False}
        if name == "set_view_mode":
            mode = args.get("mode", "3d")
            return {"ok": True, "mode": mode}, {"type": "set_view_mode",
                                                "mode": mode, "requires_confirm": False}
        if name == "report_incident":
            kind = args.get("kind", "hazard")
            # IMMEDIATE + UNDO: the client logs it after a short Undo window
            # (Gmail-style pre-commit), so an Undo tap means it never posts.
            return {"ok": True, "kind": kind, "commit": "done",
                    "note": "REPORTED (Undo on screen). Speak a short thanks/"
                            "ack; do NOT ask yes/no."}, \
                   {"type": "report_incident", "kind": kind,
                    "requires_confirm": False, "undoable": True, "commit": "done"}
        if name == "cancel_navigation":
            # The one high-stakes action → explicit confirm.
            return {"ok": True, "commit": "ask",
                    "note": "NOT ended yet. Ask ONE yes/no question to confirm "
                            "ending navigation; never say it's done."}, \
                   {"type": "cancel_navigation", "requires_confirm": True,
                    "commit": "ask"}
        return {"error": f"unknown tool {name}"}, None
    except Exception as e:
        logger.error(f"copilot tool {name} failed: {e}")
        return {"error": "tool failed"}, None

# ── OpenAI streaming ──────────────────────────────────────────────────────────
async def _stream_chat(messages: List[Dict], with_tools: bool,
                       tools: Optional[List[Dict]] = None):
    """Yield ('delta', text) and finally ('tool_calls', [...]) or ('end', None).
    Raises on transport errors; caller converts to an error line.
    `tools` overrides the schema list (CopilotV2 passes its extended set);
    None keeps the legacy _TOOLS — the legacy path is byte-identical."""
    payload: Dict[str, Any] = {
        "model": COPILOT_MODEL, "messages": messages, "stream": True,
    }
    if _GPT5_FAMILY:
        payload["max_completion_tokens"] = MAX_TOKENS
        if COPILOT_REASONING:
            payload["reasoning_effort"] = COPILOT_REASONING
    else:
        payload["max_tokens"] = MAX_TOKENS
        payload["temperature"] = COPILOT_TEMP
    if with_tools:
        payload["tools"] = tools if tools is not None else _TOOLS
        payload["tool_choice"] = "auto"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}",
               "Content-Type": "application/json"}
    tool_calls: Dict[int, Dict] = {}
    async with _client.stream("POST", CHAT_URL, headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread())[:300]
            raise RuntimeError(f"OpenAI {resp.status_code}: {body!r}")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield ("delta", delta["content"])
            for tc in (delta.get("tool_calls") or []):
                i = tc.get("index", 0)
                slot = tool_calls.setdefault(i, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
    if tool_calls:
        yield ("tool_calls", [tool_calls[k] for k in sorted(tool_calls)])
    else:
        yield ("end", None)

# ── Endpoint ──────────────────────────────────────────────────────────────────
class ConverseRequest(BaseModel):
    messages: List[Dict[str, str]] = []
    context: Optional[Dict[str, Any]] = None
    app_lang: str = "en"
    pending_action: Optional[Dict[str, Any]] = None
    # ── CopilotV2 (feature flag) ──
    # v2=True routes the turn through api/copilot_v2.py (single language
    # resolution + deterministic output validator + fact fast-path + the
    # extended tool set). Absent/false = this file's legacy path, unchanged —
    # that is the kill-switch.
    v2: bool = False
    prev_lang: Optional[str] = None   # sticky conversation language from client


@router.post("/copilot/converse")
async def copilot_converse(req: ConverseRequest):
    """Streaming copilot turn. Always returns 200 + NDJSON; failures arrive as
    an in-stream error line so the client can speak a graceful fallback."""

    if req.v2:
        from api.copilot_v2 import stream_v2   # lazy: avoids an import cycle
        return StreamingResponse(stream_v2(req),
                                 media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    async def gen() -> AsyncGenerator[str, None]:
        def line(obj: Dict) -> str:
            return json.dumps(obj, ensure_ascii=False) + "\n"

        if not OPENAI_KEY:
            yield line({"t": "error", "message": "assistant not configured"})
            return

        history = [m for m in req.messages
                   if m.get("role") in ("user", "assistant") and m.get("content")]
        history = history[-HISTORY_MAX:]
        if not history or history[-1]["role"] != "user":
            yield line({"t": "error", "message": "no user message"})
            return

        ctx = req.context or {}
        user_text = history[-1]["content"]
        lang = _detect_lang(user_text, req.app_lang)
        # Mirror the DOMINANT language of this utterance; call out code-switched
        # input explicitly so the model treats the mix as normal (never as an
        # error) and keeps brand names in their natural script. Injected as its
        # own SYSTEM message right before the user turn for strong adherence.
        if _is_mixed(user_text):
            base_rule = (
                "This user message MIXES Arabic and English — normal Egyptian "
                "code-switching, NOT an error. Reply mainly in "
                + ("Egyptian colloquial Arabic (masri, NOT فصحى)" if lang == "ar"
                   else "natural English")
                + ", keeping brand/place names in whichever script fits "
                  "naturally.")
        elif lang == "ar":
            base_rule = ("This user message is in Egyptian Arabic → reply in "
                         "Egyptian colloquial Arabic (masri, NOT فصحى).")
        else:
            base_rule = ("This user message is in English → reply in natural "
                         "English.")
        lang_rule = (
            base_rule
            + " EXCEPTION: if the user asks for a specific language (e.g. "
              "'speak Arabic', 'بالعربي', 'in English'), reply in the REQUESTED "
              "language — you are fully fluent in both and must never claim "
              "you can't speak Arabic or English.")
        ctx_block = _format_context(ctx)
        pending = ""
        if req.pending_action:
            pending = ("\n[Pending action awaiting user confirmation]\n"
                       + json.dumps(req.pending_action, ensure_ascii=False))

        messages: List[Dict[str, Any]] = [{"role": "system", "content": _SYSTEM}]
        messages += history[:-1]
        messages.append({"role": "system", "content": "LANGUAGE: " + lang_rule})
        messages.append({"role": "user",
                         "content": f"{user_text}\n\n{ctx_block}{pending}"})
        logger.info(f"copilot turn: detected_lang={lang} app_lang={req.app_lang} "
                    f"history={len(history)}")

        spoke = False           # any delta text emitted
        full_text = []          # for expects_reply heuristic
        action_out: Optional[Dict] = None

        try:
            # ── First model pass (tools enabled) ──────────────────────────────
            pending_tools: Optional[List[Dict]] = None
            async for kind, payload in _stream_chat(messages, with_tools=True):
                if kind == "delta":
                    spoke = True
                    full_text.append(payload)
                    yield line({"t": "delta", "text": payload})
                elif kind == "tool_calls":
                    pending_tools = payload

            # ── Tool pass: resolve, then stream the follow-up speech ──────────
            if pending_tools:
                tc = pending_tools[0]          # one action per turn by design
                try:
                    args = json.loads(tc["args"] or "{}")
                except Exception:
                    args = {}
                # Perceived-latency fix (reviews felt endless): speak a short
                # lead-in IMMEDIATELY for the slow tools, so the voice starts
                # while Google + the second model pass are still working.
                if tc["name"] in ("place_details", "find_places"):
                    lead = ("ثواني، بشوفلك…" if lang == "ar" else "One sec, let me check…") \
                        if tc["name"] == "place_details" else \
                        ("بدوّرلك دلوقتي…" if lang == "ar" else "Let me look that up…")
                    spoke = True
                    full_text.append(lead + " ")
                    yield line({"t": "delta", "text": lead + " "})
                result, action_out = await _execute_tool(tc["name"], args, ctx)
                # Emit the action BEFORE the follow-up speech streams, so the
                # client's map visuals (pins / preview / camera) appear while
                # the copilot is still narrating them — premium, not laggy.
                if action_out:
                    yield line({"t": "action", "action": action_out})
                messages.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": tc["id"] or "call_0", "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["args"] or "{}"}}]})
                messages.append({"role": "tool", "tool_call_id": tc["id"] or "call_0",
                                 "content": json.dumps(result, ensure_ascii=False)})
                messages.append({"role": "system",
                                 "content": _TOOL_FOLLOWUP + " " + lang_rule})
                async for kind, payload in _stream_chat(messages, with_tools=False):
                    if kind == "delta":
                        spoke = True
                        full_text.append(payload)
                        yield line({"t": "delta", "text": payload})

            if not spoke:
                fallback = ("تمام." if lang == "ar" else "Okay.")
                yield line({"t": "delta", "text": fallback})
                full_text.append(fallback)

            text = "".join(full_text).strip()
            # auto-commit counts too: the mic must reopen during the countdown
            # so a spoken "cancel" can stop the auto-add hands-free.
            expects = bool(action_out and (action_out.get("requires_confirm")
                                           or action_out.get("commit") == "auto")) \
                or text.endswith("?") or text.endswith("؟")
            yield line({"t": "done", "expects_reply": expects, "lang": lang})

        except Exception as e:
            logger.error(f"copilot converse failed: {e}")
            yield line({"t": "error", "message": "upstream failure"})

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
