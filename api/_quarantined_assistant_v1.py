"""
api/_quarantined_assistant_v1.py  (was api/assistant.py)

☠️ QUARANTINED DEAD CODE — NOT ROUTED. main.py no longer includes this router
(removed in the CopilotV2 overhaul), so /api/assistant/* returns 404. The LIVE
assistant is api/copilot.py → /api/copilot/converse (+ api/copilot_v2.py).
This file is retained only as a historical reference for the v1 pipeline shape;
do not import it, do not "fix" it, do not resurrect it.

AI assistant endpoints (all OpenAI calls are server-side, direct httpx — no SDK):
  POST /api/assistant/chat   — text in  → text reply        (phase 1, unchanged)
  POST /api/assistant/voice  — audio in → transcript + reply text + reply audio
  POST /api/assistant/tts    — text in  → reply audio (mp3, base64)

The chat-completion logic lives in ONE shared helper (_run_chat) that both
/chat and /voice call, so the system prompt + behavior stay identical.

Required env var (Cloud Run secret):
  OPENAI_API_KEY   — OpenAI secret key. Set via:
                     gcloud run services update routemind-backend \\
                       --update-secrets OPENAI_API_KEY=openai-key:latest \\
                       --region us-central1

Optional env vars (all have sane defaults):
  ASSISTANT_MODEL      — chat model.       Default "gpt-4o-mini".
  ASSISTANT_STT_MODEL  — Whisper model.    Default "whisper-1".
  ASSISTANT_TTS_MODEL  — TTS model.        Default "gpt-4o-mini-tts" (set to
                                           "tts-1" if the newer model is unavailable).
  ASSISTANT_TTS_VOICE  — TTS voice.        Default "nova" (natural for AR + EN).
"""

import os
import json
import base64
import logging
from typing import Optional, List

import httpx
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from api.places import resolve_nearby   # shared nearest-place resolver (no key in app)

logger = logging.getLogger("routemind.assistant")
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_MODEL  = os.environ.get("ASSISTANT_MODEL", "gpt-4o-mini")
WHISPER_MODEL = os.environ.get("ASSISTANT_STT_MODEL", "whisper-1")
TTS_MODEL     = os.environ.get("ASSISTANT_TTS_MODEL", "tts-1")      # faster than tts-1-hd; set tts-1-hd for crisper AR
TTS_VOICE     = os.environ.get("ASSISTANT_TTS_VOICE", "alloy")      # natural for AR+EN; try "shimmer" too
MAX_TOKENS    = 120   # short spoken replies; keeps generation fast
CHAT_TEMP     = float(os.environ.get("ASSISTANT_TEMP", "0.85"))     # higher = more natural variety, less scripted

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_STT_URL  = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_TTS_URL  = "https://api.openai.com/v1/audio/speech"

# ── Pydantic models ───────────────────────────────────────────────────────────

class TrafficSegment(BaseModel):
    road:               Optional[str]   = None
    level:              Optional[str]   = None   # "low"|"medium"|"high"|"very_high"
    distance_ahead_km:  Optional[float] = None

class CameraAhead(BaseModel):
    type:               Optional[str]   = None   # "speed"|"red_light"|"average"
    distance_ahead_km:  Optional[float] = None
    limit_kmh:          Optional[int]   = None

class TripContext(BaseModel):
    remaining_distance_km: Optional[float]            = None
    remaining_time_min:    Optional[int]              = None
    destination_name:      Optional[str]              = None
    current_road:          Optional[str]              = None
    next_maneuver:         Optional[str]              = None
    traffic_segments:      Optional[List[TrafficSegment]] = None
    cameras_ahead:         Optional[List[CameraAhead]]    = None
    speed_limit_kmh:       Optional[int]              = None
    # Current GPS — REQUIRED for the add_stop action to resolve the nearest place.
    # The app should populate these (see report: wire on the app side).
    user_lat:              Optional[float]            = None
    user_lng:              Optional[float]            = None

class ChatRequest(BaseModel):
    message:      str
    trip_context: Optional[TripContext] = None

class ChatResponse(BaseModel):
    reply:    str
    language: str   # "ar" | "en"
    # Optional structured action for the app to act on (e.g. add a stop). Null for
    # normal replies. When present, the app shows a confirm before acting.
    action:   Optional[dict] = None

class TtsRequest(BaseModel):
    text:     str
    language: Optional[str] = None   # advisory only; TTS voice handles both

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are RouteMind — a warm, sharp Egyptian companion riding shotgun in the
user's car in Cairo. You're helpful, confident, and lightly witty when it fits — like a smart
friend in the passenger seat, never a robot and never over-the-top.

═══ THE GOLDEN RULE: SOUND HUMAN, NEVER SCRIPTED ═══
VARY EVERYTHING. Never reuse the same sentence pattern, opener, or closing twice in a row.
Rotate naturally through different phrasings, word order, and length. The example phrasings
below are INSPIRATION to vary around — do NOT copy them verbatim or always pick the first one.
Two users asking the same thing should get two different-sounding answers.

═══ LANGUAGE (hard rule) ═══
A separate message tells you exactly which language the user spoke. Reply ONLY in that language.
- Arabic → EGYPTIAN COLLOQUIAL (اللهجة المصرية العامية), NEVER فصحى/MSA. Talk like a real
  Egyptian: فاضلك، يعني، حوالي، خد بالك، الطريق سايح، تمام، خلي بالك، بالتوفيق.
- English → natural, friendly English.
- Never switch or mix languages.

═══ LENGTH & SAFETY ═══
1–2 short sentences, always — it's read aloud while driving. Calm, clear, non-distracting.

═══ PERSONALITY BY SITUATION (vary the wording every time) ═══
GREETINGS (match the time of day, rotate):
  AR: «صباح الخير» / «مساء الخير» / «أهلاً» / «إزيك، عامل إيه؟»
  EN: "Good morning" / "Good evening" / "Hey there" / "Hey, how's it going?"

SIGN-OFFS (rotate, never the same one twice):
  AR: «وصول آمن» / «خلي بالك من نفسك» / «بالتوفيق» / «تسلم» / «سلامة الوصول»
  EN: "Drive safe" / "Take care out there" / "Safe travels" / "You got this"

TRAFFIC (read traffic_segments; tailor tone to severity, summarize smartly):
  • heavy/very_high close ahead → mild heads-up:
    AR: «خد بالك، في زحمة تقيلة بعد X كيلو» EN: "Heads up — heavy traffic about X km ahead."
  • moderate → reassure: AR: «في تباطؤ بسيط بعد X بس ماشي» EN: "A bit slow around X km, but it's moving."
  • clear (empty array) → upbeat & varied:
    AR: «الطريق سايح قدامك، مفيش زحمة» / «صافي كله، مفيش حاجة قدامك»
    EN: "Road's clear ahead, smooth sailing" / "All clear up front."
  • multiple segments → combine intelligently using their distances + levels, e.g.
    AR: «تباطؤ بسيط بعد ٨ كيلو، وكمان زحمة حوالي ٤٠» EN: "Light slowdown in 8 km, then heavier near 40."
  • road names are NOT available (Mapbox limitation): NEVER say a flat "I don't have info."
    If asked for the road's name, say you can't see exact names but give the distance + severity instead:
    AR: «مش باين اسم الطريق، بس في زحمة تقيلة بعد ٣ كيلو» EN: "Can't see the exact road name, but there's heavy traffic ~3 km up."

CAMERAS (read cameras_ahead):
  • present → warn with distance + limit, varied:
    AR: «كاميرا سرعة بعد ١.٢ كيلو، الحد ٩٠» EN: "Speed camera in 1.2 km, limit 90."
  • empty → natural & varied: AR: «مفيش كاميرات قدامك دلوقتي» EN: "No cameras coming up."

TIME / DISTANCE (contextualize, don't just read numbers):
  • far (≥1h): AR: «لسه بدري، استريح — فاضلك حوالي ساعة» EN: "Still a way to go, relax — about an hour left."
  • close: AR: «خلاص قربت، فاضلك شوية» EN: "Almost there, just a little left."

GENERAL / OFF-TOPIC (be a real assistant): answer ANY question — Egypt geography, landmarks,
driving tips, light chit-chat, even a joke if asked — briefly, in the same warm voice, using
your general knowledge freely. Play along with fun, stay friendly.

SPECIAL CASES:
  • garbled/unclear → politely ask to repeat, varied: AR: «معلش مسمعتش كويس، ممكن تعيد؟» EN: "Sorry, didn't catch that — say again?"
  • data genuinely missing → graceful, say what you DO know; never a canned refusal.
  • thanks → warm & varied: AR: «العفو يا فندم» / «تحت أمرك» / «دي مهمتي» EN: "Anytime" / "You got it" / "Happy to help."
  • stressed/rushed user → reassure calmly and briefly.

═══ ACCURACY ═══
Use ONLY the trip context for navigation facts. NEVER invent road names, camera counts,
distances, or times that aren't in the context — but phrase any gap gracefully (see above),
never robotically. For general knowledge, your own knowledge is fine."""

# ── Action tools (function-calling) ─────────────────────────────────────────────

# ONE tool for now: add the nearest place of a category as a stop.
ADD_STOP_TOOL = {
    "type": "function",
    "function": {
        "name": "add_stop",
        "description": (
            "Add the nearest place of a given category as a stop on the user's current "
            "navigation route. Use when the user asks to add/stop at a gas station, "
            "pharmacy, ATM, restaurant, or cafe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["gas_station", "pharmacy", "atm", "restaurant", "cafe"],
                },
            },
            "required": ["category"],
        },
    },
}

# Second-pass nudge: after the lookup runs, produce the spoken confirm question.
_ADD_STOP_FOLLOWUP = (
    "You called add_stop and just received the lookup result. Reply in the user's "
    "language, 1–2 short warm sentences: if found=true, say the place name and its "
    "approximate distance (round to km when ≥1000 m) and ASK if they want to add it "
    "(they'll confirm with a tap). If found=false with reason 'no_result', say you "
    "couldn't find one nearby. If reason 'no_location', say you can't tell where they "
    "are right now. Vary your wording naturally."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context_block(ctx: Optional[TripContext]) -> str:
    """Serialize TripContext into a compact English text block appended to the user message.
    English is used here regardless of the user's language — the model reads this as data."""
    if ctx is None:
        return ""

    lines: List[str] = []

    if ctx.remaining_distance_km is not None:
        lines.append(f"Remaining distance: {ctx.remaining_distance_km:.1f} km")
    if ctx.remaining_time_min is not None:
        lines.append(f"Remaining time: {ctx.remaining_time_min} min")
    if ctx.destination_name:
        lines.append(f"Destination: {ctx.destination_name}")
    if ctx.current_road:
        lines.append(f"Current road: {ctx.current_road}")
    if ctx.next_maneuver:
        lines.append(f"Next maneuver: {ctx.next_maneuver}")
    if ctx.speed_limit_kmh is not None:
        lines.append(f"Speed limit: {ctx.speed_limit_kmh} km/h")

    if ctx.traffic_segments:
        segs = []
        for s in ctx.traffic_segments:
            parts = []
            if s.road:                    parts.append(s.road)
            if s.level:                   parts.append(f"traffic={s.level}")
            if s.distance_ahead_km is not None:
                parts.append(f"{s.distance_ahead_km:.1f} km ahead")
            segs.append(", ".join(parts))
        lines.append("Traffic segments: " + " | ".join(segs))

    if ctx.cameras_ahead:
        cams = []
        for c in ctx.cameras_ahead:
            parts = []
            if c.type:                    parts.append(c.type)
            if c.distance_ahead_km is not None:
                parts.append(f"{c.distance_ahead_km:.1f} km ahead")
            if c.limit_kmh is not None:   parts.append(f"limit {c.limit_kmh} km/h")
            cams.append(", ".join(parts))
        lines.append("Cameras ahead: " + " | ".join(cams))

    if not lines:
        return ""
    return "\n\n[Current trip data]\n" + "\n".join(lines)


def _detect_language(text: str) -> str:
    """Heuristic: any Arabic Unicode codepoint → 'ar', otherwise 'en'."""
    return "ar" if any('؀' <= ch <= 'ۿ' for ch in text) else "en"


def _fallback(lang: str, reason: str = "error") -> ChatResponse:
    """Friendly fallback reply when OpenAI is unavailable."""
    if lang == "ar":
        msg = {
            "error":     "فيه مشكلة مع المساعد، جرب تاني.",
            "timeout":   "المساعد بطيء دلوقتي، جرب تاني.",
            "no_key":    "المساعد مش مفعّل دلوقتي.",
            "no_pkg":    "المساعد مش متاح دلوقتي.",
        }.get(reason, "فيه مشكلة، جرب تاني.")
    else:
        msg = {
            "error":     "Assistant hit an error, please try again.",
            "timeout":   "I'm a bit slow right now, please try again.",
            "no_key":    "Assistant is not configured.",
            "no_pkg":    "Assistant is unavailable right now.",
        }.get(reason, "Something went wrong, please try again.")
    return ChatResponse(reply=msg, language=lang)


def _voice_error_text(lang: str, kind: str) -> str:
    """Short spoken-friendly error line for the voice flow (no chat reply available)."""
    ar = {
        "empty":      "مسمعتش حاجة واضحة، جرب تاني.",
        "stt_failed": "مقدرتش أفهم الصوت، جرب تاني.",
    }
    en = {
        "empty":      "I didn't catch that, please try again.",
        "stt_failed": "I couldn't understand the audio, please try again.",
    }
    table = ar if lang == "ar" else en
    return table.get(kind, table["stt_failed"])


async def _run_chat(message: str, trip_context: Optional[TripContext]) -> ChatResponse:
    """
    Shared chat-completion logic for BOTH /chat and /voice.

    Function-calling enabled with ONE tool (add_stop). Normal questions return a
    reply with action=None (a single model call). When the model calls add_stop,
    we resolve the nearest place via the shared resolve_nearby() and do a second
    model call so the spoken reply naturally asks the user to confirm; the
    structured action is returned alongside. Never raises — friendly fallback.
    """
    # EXPLICIT language detection on the user's own message (not left to the model).
    lang = _detect_language(message)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY env var is not set")
        return _fallback(lang, "no_key")

    ctx_block    = _build_context_block(trip_context)
    user_content = message + ctx_block

    # Hard, per-request language directive so the model can't infer/guess language.
    lang_name = "Egyptian Arabic (اللهجة المصرية العامية, NOT MSA/فصحى)" \
                if lang == "ar" else "English"
    lang_rule = (f"The user spoke {lang_name}. Reply ONLY in {lang_name}. "
                 f"Do not use any other language under any circumstances.")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "system", "content": lang_rule},
        {"role": "user",   "content": user_content},
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    async def _call(client: "httpx.AsyncClient", msgs: list, with_tools: bool) -> dict:
        payload = {
            "model":       OPENAI_MODEL,
            "messages":    msgs,
            "max_tokens":  MAX_TOKENS,
            "temperature": CHAT_TEMP,   # variety so replies never sound scripted
        }
        if with_tools:
            payload["tools"]       = [ADD_STOP_TOOL]
            payload["tool_choice"] = "auto"
        r = await client.post(OPENAI_CHAT_URL, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _call(client, messages, with_tools=True)
            msg  = data["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []

            # ── Normal Q&A (no tool) — unchanged behavior ────────────────────
            if not tool_calls:
                reply_text = (msg.get("content") or "").strip()
                if not reply_text:
                    return _fallback(lang, "error")
                reply_lang = _detect_language(reply_text)
                logger.info(f"assistant_chat: in_lang={lang} reply_lang={reply_lang} action=none")
                return ChatResponse(reply=reply_text, language=reply_lang, action=None)

            # ── Tool call(s) → resolve add_stop, then 2nd call for the confirm line ──
            action: Optional[dict] = None
            tool_msgs: list = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                if fn.get("name") == "add_stop" and action is None:
                    try:
                        cat = (json.loads(fn.get("arguments") or "{}") or {}).get("category")
                    except Exception:
                        cat = None
                    lat = getattr(trip_context, "user_lat", None) if trip_context else None
                    lng = getattr(trip_context, "user_lng", None) if trip_context else None

                    results = []
                    if cat and lat is not None and lng is not None:
                        results = await resolve_nearby(lat, lng, category=cat, limit=1)

                    if results:
                        pl = results[0]
                        action = {
                            "type": "add_stop", "name": pl.name,
                            "lat": pl.lat, "lng": pl.lng, "address": pl.address,
                            "distance_m": pl.distance_m, "needs_confirm": True,
                        }
                        tool_result = {"found": True, "name": pl.name,
                                       "distance_m": pl.distance_m, "category": cat}
                    else:
                        reason = "no_location" if (lat is None or lng is None) else "no_result"
                        tool_result = {"found": False, "reason": reason, "category": cat}
                    tool_msgs.append({"role": "tool", "tool_call_id": tc.get("id"),
                                      "content": json.dumps(tool_result, ensure_ascii=False)})
                else:
                    # Every tool_call MUST get a tool result before the next turn.
                    tool_msgs.append({"role": "tool", "tool_call_id": tc.get("id"),
                                      "content": json.dumps({"error": "unsupported"})})

            assistant_turn = {"role": "assistant", "content": msg.get("content"),
                              "tool_calls": tool_calls}
            followup = messages + [assistant_turn] + tool_msgs + [
                {"role": "system", "content": _ADD_STOP_FOLLOWUP},
            ]
            data2 = await _call(client, followup, with_tools=False)
            reply_text = (data2["choices"][0]["message"].get("content") or "").strip()
            if not reply_text:
                reply_text = _fallback(lang, "error").reply
            reply_lang = _detect_language(reply_text)
            logger.info(f"assistant_chat: in_lang={lang} reply_lang={reply_lang} "
                        f"action={'add_stop' if action else 'add_stop_none'}")
            return ChatResponse(reply=reply_text, language=reply_lang, action=action)

    except httpx.TimeoutException:
        logger.warning("OpenAI request timed out")
        return _fallback(lang, "timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI HTTP error {e.response.status_code}: {e.response.text[:200]}")
        return _fallback(lang, "error")
    except Exception as e:
        logger.error(f"Unexpected assistant error: {e}")
        return _fallback(lang, "error")


async def _transcribe_audio(
    api_key: str,
    audio_bytes: bytes,
    filename: Optional[str],
    content_type: Optional[str],
    language_hint: Optional[str],
) -> str:
    """Transcribe audio with OpenAI Whisper (multipart, direct httpx). Returns the
    spoken text. Raises httpx errors on failure (caller handles gracefully)."""
    files = {
        "file": (
            filename or "audio.m4a",
            audio_bytes,
            content_type or "application/octet-stream",
        ),
    }
    data = {"model": WHISPER_MODEL, "response_format": "json"}
    if language_hint in ("ar", "en"):
        data["language"] = language_hint   # improves accuracy, esp. for Arabic

    # Whisper sets the multipart Content-Type (with boundary) itself — don't set it.
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            OPENAI_STT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
        )
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()


async def _synthesize_speech(api_key: str, text: str) -> Optional[bytes]:
    """Generate spoken mp3 of `text` with OpenAI TTS (direct httpx). Returns the
    mp3 bytes, or None on any failure (caller falls back to device TTS / null)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENAI_TTS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":           TTS_MODEL,
                    "voice":           TTS_VOICE,
                    "input":           text,
                    "response_format": "mp3",
                },
            )
            resp.raise_for_status()
            return resp.content
    except httpx.HTTPStatusError as e:
        logger.error(f"TTS HTTP error {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
    return None

# ── Endpoints ───────────────────────────────────────────────────────────────────

@router.post("/assistant/chat", response_model=ChatResponse)
async def assistant_chat(body: ChatRequest):
    """
    Text-in / text-out assistant (unchanged behavior). Thin wrapper over the
    shared _run_chat() helper that /voice also uses.
    """
    return await _run_chat(body.message, body.trip_context)


@router.post("/assistant/voice")
async def assistant_voice(
    audio:         UploadFile      = File(...),
    trip_context:  Optional[str]   = Form(None),   # JSON string of TripContext
    language_hint: Optional[str]   = Form(None),   # "ar" | "en"
):
    """
    Voice assistant — audio in, (transcript + reply text + reply audio) out.

    Flow (all server-side, OPENAI_API_KEY never leaves the backend):
      1. Whisper transcribes the audio  → user's spoken text
      2. _run_chat() (same logic as /chat) → reply text + detected language
      3. OpenAI TTS speaks the reply     → mp3, returned base64

    Always returns HTTP 200 with JSON so the app can parse reliably:
      { "transcript", "reply", "language", "audio_base64", ["error"] }
    On TTS failure, audio_base64 is null and the app falls back to device TTS.
    """
    hint     = language_hint if language_hint in ("ar", "en") else None
    err_lang = hint or "en"

    # ── Guard: API key ───────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY env var is not set")
        return {
            "transcript": "", "reply": _fallback(err_lang, "no_key").reply,
            "language": err_lang, "audio_base64": None, "error": "no_api_key",
        }

    # ── Read audio ───────────────────────────────────────────────────────────
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        logger.error(f"voice: could not read upload: {e}")
        audio_bytes = b""
    if not audio_bytes:
        return {
            "transcript": "", "reply": _voice_error_text(err_lang, "empty"),
            "language": err_lang, "audio_base64": None, "error": "empty_audio",
        }

    # ── Parse optional trip_context (ignore if malformed) ────────────────────
    parsed_ctx: Optional[TripContext] = None
    if trip_context:
        try:
            parsed_ctx = TripContext(**json.loads(trip_context))
        except Exception as e:
            logger.warning(f"voice: ignoring malformed trip_context: {e}")

    # ── 1. Transcribe (Whisper) ──────────────────────────────────────────────
    try:
        transcript = await _transcribe_audio(
            api_key, audio_bytes, audio.filename, audio.content_type, hint)
    except httpx.TimeoutException:
        logger.warning("Whisper request timed out")
        return {
            "transcript": "", "reply": _voice_error_text(err_lang, "stt_failed"),
            "language": err_lang, "audio_base64": None, "error": "transcription_timeout",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"Whisper HTTP error {e.response.status_code}: {e.response.text[:200]}")
        return {
            "transcript": "", "reply": _voice_error_text(err_lang, "stt_failed"),
            "language": err_lang, "audio_base64": None, "error": "transcription_failed",
        }
    except Exception as e:
        logger.error(f"Whisper failed: {e}")
        return {
            "transcript": "", "reply": _voice_error_text(err_lang, "stt_failed"),
            "language": err_lang, "audio_base64": None, "error": "transcription_failed",
        }

    if not transcript:
        return {
            "transcript": "", "reply": _voice_error_text(err_lang, "empty"),
            "language": err_lang, "audio_base64": None, "error": "empty_transcript",
        }

    # ── 2. Chat (shared helper — identical to /chat behavior) ─────────────────
    chat = await _run_chat(transcript, parsed_ctx)

    # ── 3. TTS the reply (graceful: null audio on failure) ────────────────────
    audio_b64 = None
    mp3 = await _synthesize_speech(api_key, chat.reply)
    if mp3:
        audio_b64 = base64.b64encode(mp3).decode("ascii")

    logger.info(
        f"assistant_voice: lang={chat.language} transcript_len={len(transcript)} "
        f"tts={'ok' if audio_b64 else 'null'} mp3_bytes={len(mp3) if mp3 else 0}"
    )

    return {
        "transcript":   transcript,
        "reply":        chat.reply,
        "language":     chat.language,
        "audio_base64": audio_b64,
        "action":       chat.action,   # null for normal replies; add_stop action otherwise
    }


@router.post("/assistant/tts")
async def assistant_tts(body: TtsRequest):
    """
    Text → speech. Returns { "audio_base64": "<mp3>" } or, on failure,
    { "audio_base64": null, "error": "..." } (HTTP 200, app falls back).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY env var is not set")
        return {"audio_base64": None, "error": "no_api_key"}

    text = (body.text or "").strip()
    if not text:
        return {"audio_base64": None, "error": "empty_text"}

    mp3 = await _synthesize_speech(api_key, text)
    if not mp3:
        return {"audio_base64": None, "error": "tts_failed"}

    return {"audio_base64": base64.b64encode(mp3).decode("ascii")}
