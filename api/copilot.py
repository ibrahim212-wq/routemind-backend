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
import re
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
# The system prompt lives in api/copilot_v2.py (SYSTEM_V2). This module keeps
# the tool schemas, the tool executors, the context helpers and the OpenAI
# streaming client that the v2 engine shares.

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


_MAX_STOP_DETOUR_MIN = 120      # beyond this a "stop" is a different trip
_MAX_STOP_DISTANCE_KM = 80
_LITERAL_STOP = {"stop", "a stop", "the stop", "stops", "new stop", "a new stop",
                 "وقفه", "وقفة", "الوقفه", "الوقفة", "وقفه جديده", "وقفة جديدة",
                 "محطه", "محطة", "استوب", "ستوب", "وقوف"}


def _is_literal_stop(s: str) -> bool:
    t = (s or "").strip().lower()
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي"), ("ـ", "")):
        t = t.replace(a, b)
    return t in _LITERAL_STOP


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
            # Guardrail 0 — "add a stop" / «ضيف وقفة» carries no place and no
            # category. Searching the literal word found "stop and shop" and a
            # "Waqfa Cafe" in Saudi Arabia (2026-09 probe). Ask instead.
            if _is_literal_stop(qname) or _is_literal_stop(query) or not (qname or query):
                return {"found": False, "no_category": True,
                        "note": "No place or category was given. Ask ONE short "
                                "question: a stop where — gas, pharmacy, food?"}, None
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
            # Guardrail 3 — a match that is nowhere near this trip (the text
            # search is biased, not bounded) is not a stop: say so, don't
            # preview a +3177-minute detour.
            far_km = (place.get("distance_m") or 0) / 1000.0
            if (added is not None and added > _MAX_STOP_DETOUR_MIN) or far_km > _MAX_STOP_DISTANCE_KM:
                return {"found": False, "requested": qname or query, "too_far": True,
                        "note": "The only match is far outside this trip — say it "
                                "isn't on the way and ask for a closer type or "
                                "place. Never invent a nearer one."}, None
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
    `tools` overrides the schema list (CopilotV2 passes its extended set).

    429 (tokens-per-minute) is retried with the delay OpenAI itself suggests
    ("try again in 976ms"), capped, up to _RATE_LIMIT_RETRIES times — a
    rate-limited turn used to fail outright and the driver heard an error."""
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
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        async with _client.stream("POST", CHAT_URL, headers=headers, json=payload) as resp:
            if resp.status_code == 429 and attempt < _RATE_LIMIT_RETRIES:
                body = (await resp.aread()).decode("utf-8", "ignore")
                delay = _retry_after_s(body)
                logger.warning(f"OpenAI 429 — retrying in {delay:.2f}s "
                               f"(attempt {attempt + 1}/{_RATE_LIMIT_RETRIES})")
                await asyncio.sleep(delay)
                continue
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
            break
    if tool_calls:
        yield ("tool_calls", [tool_calls[k] for k in sorted(tool_calls)])
    else:
        yield ("end", None)


_RATE_LIMIT_RETRIES = 2
_RETRY_AFTER_RE = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b")


def _retry_after_s(body: str) -> float:
    """OpenAI's own suggestion ('try again in 976ms'), capped so a turn never
    silently waits past the client's watchdog."""
    m = _RETRY_AFTER_RE.search(body or "")
    if not m:
        return 1.0
    v = float(m.group(1))
    secs = v / 1000.0 if m.group(2) == "ms" else v
    return max(0.2, min(secs + 0.15, 2.5))


# ── Endpoint ──────────────────────────────────────────────────────────────────
class ConverseRequest(BaseModel):
    messages: List[Dict[str, str]] = []
    context: Optional[Dict[str, Any]] = None
    app_lang: str = "en"                 # the app-UI language: the LAST fallback only
    pending_action: Optional[Dict[str, Any]] = None
    prev_lang: Optional[str] = None      # the conversation's resolved language (sticky)
    # What produced the transcript — lets the resolver refuse a language
    # switch that is really a wrong-language microphone (see copilot_lang).
    stt_lang: Optional[str] = None       # "ar" | "en": the recognizer that heard it
    stt_confidence: Optional[float] = None


@router.post("/copilot/converse")
async def copilot_converse(req: ConverseRequest):
    """Streaming copilot turn. Always returns 200 + NDJSON; failures arrive as
    an in-stream error line (with a localized `spoken` text) so the client can
    speak a graceful fallback in the conversation's language.

    There is ONE turn engine (api/copilot_v2.py). The former legacy generator
    — un-validated, no language resolution — was deleted: every client build,
    old or new, now gets the gated engine. (Unknown request fields such as
    the old `v2` flag are ignored by pydantic.)"""
    from api.copilot_v2 import stream_v2   # lazy: avoids an import cycle
    return StreamingResponse(stream_v2(req),
                             media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
