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
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.places import resolve_nearby, _google_corridor_pois, _filter_to_corridor, corridor_anchors
from services import route_geometry as geo

logger = logging.getLogger("routemind.copilot")
router = APIRouter()

# ── Config (env-overridable) ──────────────────────────────────────────────────
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
CHAT_URL       = "https://api.openai.com/v1/chat/completions"
COPILOT_MODEL  = os.getenv("COPILOT_MODEL", "gpt-4o-mini")   # fast + strong tools
COPILOT_TEMP   = float(os.getenv("COPILOT_TEMP", "0.8"))
MAX_TOKENS     = int(os.getenv("COPILOT_MAX_TOKENS", "180"))
HISTORY_MAX    = int(os.getenv("COPILOT_HISTORY_MAX", "24"))  # turns kept per request
POI_LIMIT      = 5                                            # per show_pois resolve

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
- Round numbers like a human ("about ten minutes", "two kilometers").

LANGUAGE — you are FULLY BILINGUAL (Egyptian Arabic + English):
- Mirror the user's language per message: Arabic in → Egyptian colloquial
  Arabic out (masri, NOT فصحى). English in → natural English out.
- If the user asks for a specific language ("speak Arabic", "بالعربي"), switch
  to it and stay in it. NEVER claim you cannot speak Arabic or English — you
  speak both natively.

TRUTH — never fabricate, ever:
- For anything about the trip (roads, ETA, traffic, cameras, alternatives,
  stops, distances) use ONLY the [Current trip data] block. NEVER invent or
  estimate a number, road name, or condition that isn't in the data — a
  made-up "traffic is moderate" is a serious failure.
- Read the data precisely: "Traffic ahead: none detected" means the road is
  clear — say that confidently. A missing field means you genuinely don't
  have that data — say so honestly in one short sentence, and offer what you
  DO know instead (e.g. the ETA) or ask a clarifying question.
- INTENT PRECISION: "nearest X to me / near me / around here" → mode='nearest'
  (closest to the user's location, ignore the route). "on my route / on the
  way / ahead" → mode='route'. Get this right — they are different searches.
- General questions (history, food, football, anything) — answer them! Briefly,
  and when natural, tie back to the drive.

ACTIONS — you can DO things, not just talk:
- Use your tools whenever the user wants something done or shown: finding
  places, adding stops, alternatives, rerouting via a road, camera views,
  muting guidance, reporting incidents, ending navigation.
- When the user's request is clear (e.g. "add the nearest gas station"), call
  the tool RIGHT AWAY — do not ask a clarifying question first. Asking, then
  calling the tool, then asking AGAIN is wrong: it double-prompts the user.
- Mutating actions (add stop, switch route, reroute, report, end navigation)
  are PROPOSALS. After the tool previews it on the map, say in ONE short
  sentence what you're about to do and ask ONE yes/no question. That is the
  ONLY confirmation you give — a Confirm/Cancel button is already on screen,
  so do not ask twice or in two different ways.
- NEVER speak as if a mutating action is already done. Do NOT say "I've added
  it / done / added the stop" — nothing is applied until the user confirms.
  Use future/proposal wording only ("want me to add it?", "أضيفها؟").
- View-only actions (showing places, alternatives, overview, zoom) happen
  immediately — narrate what's now on screen, then offer the logical next
  step as a question.
- If a pending action is noted and the user's reply refines it ("the second
  one", "no, the cafe instead"), call the tool again with the refinement.

MEMORY: the conversation history is this trip's shared memory — refer back to
it naturally (stops already added, things already discussed)."""

_TOOL_FOLLOWUP = (
    "Tool result received. Reply in the user's language, 1–2 short spoken "
    "sentences. If the tool found something, name it with its distance. If the "
    "result note says the action needs confirmation, ask exactly ONE short "
    "yes/no question and DO NOT claim it's done — it is only a preview until "
    "the user confirms (never say 'added' / 'done' / 'اتضافت'). If it found "
    "nothing, say so plainly and suggest the closest alternative move."
)

# ── Tools (OpenAI function-calling schema) ────────────────────────────────────
_CATEGORIES = ["fuel", "restaurant", "cafe", "atm", "parking", "pharmacy"]

def _tool(name: str, desc: str, props: Dict[str, Any], required: List[str]) -> Dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

_TOOLS: List[Dict] = [
    _tool("show_pois",
          "Find and display places of a category (pins on the map). Use when "
          "the user asks to see/find gas stations, restaurants, cafes, ATMs, "
          "parking or pharmacies. mode='route' (default) = along the remaining "
          "route; mode='nearest' = closest to the user's CURRENT location "
          "regardless of route (use when they say 'near me' / 'nearest to me').",
          {"category": {"type": "string", "enum": _CATEGORIES},
           "mode": {"type": "string", "enum": ["route", "nearest"]}}, ["category"]),
    _tool("add_stop",
          "Propose adding ONE stop to the route (requires user confirmation). "
          "The nearest of a category (along the route, or nearest to the user "
          "if mode='nearest'), or a specific named place.",
          {"category": {"type": "string", "enum": _CATEGORIES},
           "mode": {"type": "string", "enum": ["route", "nearest"]},
           "place_name": {"type": "string",
                          "description": "Specific place name if the user named one"}},
          []),
    _tool("show_alternatives",
          "Display the alternative routes on the map with their time deltas. "
          "Use the alternatives in trip data to recommend the best.", {}, []),
    _tool("switch_route",
          "Propose switching to an alternative route (requires confirmation).",
          {"index": {"type": "integer",
                     "description": "1-based index from the alternatives list"}},
          ["index"]),
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
    _tool("cancel_navigation",
          "End the current navigation session (requires confirmation).", {}, []),
]

# Which action types the CLIENT must confirm before executing.
_CONFIRM_REQUIRED = {"add_stop", "switch_route", "reroute_via",
                     "report_incident", "cancel_navigation"}

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
            km = s.get("distance_km", s.get("distance_ahead_km", "?"))
            return f"{s.get('road','road')}: {s.get('level','?')} at {km} km"
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
        L.append("Alternative routes: " + " | ".join(
            f"#{a.get('index','?')}: {a.get('desc','')} ({a.get('delta_min','?')} min vs current)"
            if isinstance(a, dict) else str(a) for a in alts[:3]))
    add("Map view", "view_mode")
    if ctx.get("voice_muted") is not None:
        L.append(f"Guidance voice muted: {ctx['voice_muted']}")
    add("Local time", "local_time")
    return "\n".join(L)


def _detect_lang(text: str, app_lang: str) -> str:
    """Arabic Unicode presence → ar (same proven check as assistant v1)."""
    if any("؀" <= c <= "ۿ" for c in text):
        return "ar"
    if any("a" <= c.lower() <= "z" for c in text):
        return "en"
    return app_lang if app_lang in ("ar", "en") else "en"

# ── Server-side action resolvers (reuse the Add-Stop machinery) ───────────────
def _route_pts(ctx: Dict[str, Any]):
    """Context route [[lng,lat],...] → [(lat,lng),...]; [] when absent."""
    return geo.geojson_to_latlng(ctx.get("route") or [])


async def _resolve_pois(ctx: Dict[str, Any], category: str,
                        limit: int = POI_LIMIT, nearest: bool = False) -> List[Dict]:
    """Category POIs. nearest=False → along the remaining route (corridor search,
    same engine as /along-route/pois). nearest=True → closest to the user's
    current location regardless of route (circle search via resolve_nearby)."""
    gtype = {"fuel": "gas_station", "restaurant": "restaurant", "cafe": "cafe",
             "atm": "atm", "parking": "parking", "pharmacy": "pharmacy"}.get(category)
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


async def _resolve_place(ctx: Dict[str, Any], query: str) -> Optional[Dict]:
    """Free-text place/road resolve near the user (Google text search)."""
    lat = ctx.get("user_lat"); lng = ctx.get("user_lng")
    if lat is None or lng is None or not query.strip():
        return None
    results = await resolve_nearby(lat, lng, query=query, limit=1)
    if not results:
        return None
    r = results[0]
    return {"name": r.name, "lat": r.lat, "lng": r.lng,
            "address": r.address, "distance_m": r.distance_m}


async def _execute_tool(name: str, args: Dict[str, Any],
                        ctx: Dict[str, Any]) -> (Dict, Optional[Dict]):
    """Run a tool. Returns (tool_result_for_model, action_for_client|None)."""
    try:
        if name == "show_pois":
            cat = args.get("category", "fuel")
            nearest = args.get("mode") == "nearest"
            pois = await _resolve_pois(ctx, cat, nearest=nearest)
            if not pois:
                return {"found": False, "category": cat}, None
            dist_key = "km_from_you" if nearest else "km_ahead_on_route"
            brief = [{"name": p["name"],
                      dist_key: round(p.get("along_route_distance_m", 0) / 1000.0, 1)}
                     for p in pois]
            action = {"type": "show_pois", "category": cat, "pois": pois,
                      "mode": "nearest" if nearest else "route", "requires_confirm": False}
            where = ("straight-line from the user's location (NOT on the route)"
                     if nearest else "ahead along the route")
            return {"found": True, "category": cat, "places": brief,
                    "distances_are": where,
                    "note": "Places are now shown on the map. Phrase distances "
                            "accordingly."}, action

        if name == "add_stop":
            cat = args.get("category"); qname = args.get("place_name")
            nearest = args.get("mode") == "nearest"
            place = None
            if qname:
                place = await _resolve_place(ctx, qname)
            elif cat:
                pois = await _resolve_pois(ctx, cat, limit=1, nearest=nearest)
                if pois:
                    p = pois[0]
                    place = {"name": p["name"], "lat": p["lat"], "lng": p["lng"],
                             "distance_m": p.get("along_route_distance_m", 0)}
            if not place:
                return {"found": False}, None
            action = {"type": "add_stop", "place": place, "requires_confirm": True}
            dist_key = "km_from_you" if (nearest and not qname) else "km_ahead_on_route"
            return {"found": True, "place": place["name"],
                    dist_key: round(place.get("distance_m", 0) / 1000.0, 1),
                    "note": "PREVIEW ONLY — nothing added yet. Ask ONE yes/no "
                            "question to confirm; never say it's done. Phrase the "
                            "distance per the key name (from you vs on route)."}, action

        if name == "show_alternatives":
            alts = ctx.get("alternatives") or []
            action = {"type": "show_alternatives", "requires_confirm": False}
            if not alts:
                return {"found": False,
                        "note": "No alternative routes available right now."}, action
            return {"found": True, "alternatives": alts,
                    "note": "Alternatives now displayed. Recommend the best."}, action

        if name == "switch_route":
            idx = int(args.get("index", 1))
            action = {"type": "switch_route", "index": idx, "requires_confirm": True}
            return {"ok": True, "index": idx,
                    "note": "PREVIEW ONLY — not switched yet. Ask ONE yes/no "
                            "question; never say it's done."}, action

        if name == "reroute_via":
            via = args.get("via", "")
            place = await _resolve_place(ctx, via)
            if not place:
                return {"found": False, "via": via}, None
            action = {"type": "reroute_via", "point": place, "requires_confirm": True}
            return {"found": True, "via": place["name"],
                    "note": "PREVIEW ONLY — not rerouted yet. Ask ONE yes/no "
                            "question; never say it's done."}, action

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
            return {"ok": True, "kind": kind,
                    "note": "NOT reported yet. Ask ONE yes/no question; never "
                            "say it's done."}, \
                   {"type": "report_incident", "kind": kind, "requires_confirm": True}
        if name == "cancel_navigation":
            return {"ok": True, "note": "NOT ended yet. Ask ONE yes/no question "
                    "to confirm ending navigation; never say it's done."}, \
                   {"type": "cancel_navigation", "requires_confirm": True}
        return {"error": f"unknown tool {name}"}, None
    except Exception as e:
        logger.error(f"copilot tool {name} failed: {e}")
        return {"error": "tool failed"}, None

# ── OpenAI streaming ──────────────────────────────────────────────────────────
async def _stream_chat(messages: List[Dict], with_tools: bool):
    """Yield ('delta', text) and finally ('tool_calls', [...]) or ('end', None).
    Raises on transport errors; caller converts to an error line."""
    payload: Dict[str, Any] = {
        "model": COPILOT_MODEL, "messages": messages, "stream": True,
        "temperature": COPILOT_TEMP, "max_tokens": MAX_TOKENS,
    }
    if with_tools:
        payload["tools"] = _TOOLS
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


@router.post("/copilot/converse")
async def copilot_converse(req: ConverseRequest):
    """Streaming copilot turn. Always returns 200 + NDJSON; failures arrive as
    an in-stream error line so the client can speak a graceful fallback."""

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
        # ITEM 1 FIX — the old rule was an absolute prohibition ("Reply ONLY in
        # English. Do not use any other language") so an English request like
        # "speak Arabic" made the model literally REFUSE ("I can't speak Arabic").
        # New rule: default = mirror the detected language, with an explicit
        # exception honoring a requested language. Injected as its own SYSTEM
        # message right before the user turn (stronger adherence than the old
        # parenthetical buried after the English context block).
        lang_rule = (
            ("This user message is in Egyptian Arabic → reply in Egyptian "
             "colloquial Arabic (masri, NOT فصحى)." if lang == "ar" else
             "This user message is in English → reply in natural English.")
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
            expects = bool(action_out and action_out.get("requires_confirm")) \
                or text.endswith("?") or text.endswith("؟")
            yield line({"t": "done", "expects_reply": expects, "lang": lang})

        except Exception as e:
            logger.error(f"copilot converse failed: {e}")
            yield line({"t": "error", "message": "upstream failure"})

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
