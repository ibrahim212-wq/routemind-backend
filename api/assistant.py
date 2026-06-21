"""
api/assistant.py

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

logger = logging.getLogger("routemind.assistant")
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_MODEL  = os.environ.get("ASSISTANT_MODEL", "gpt-4o-mini")
WHISPER_MODEL = os.environ.get("ASSISTANT_STT_MODEL", "whisper-1")
TTS_MODEL     = os.environ.get("ASSISTANT_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE     = os.environ.get("ASSISTANT_TTS_VOICE", "nova")
MAX_TOKENS    = 150   # short spoken replies; bump for richer responses if needed

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

class ChatRequest(BaseModel):
    message:      str
    trip_context: Optional[TripContext] = None

class ChatResponse(BaseModel):
    reply:    str
    language: str   # "ar" | "en"

class TtsRequest(BaseModel):
    text:     str
    language: Optional[str] = None   # advisory only; TTS voice handles both

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are RouteMind, a friendly in-car navigation assistant for Cairo, Egypt.

Critical rules — follow ALL of them without exception:
1. LANGUAGE: Reply in the EXACT same language and dialect the user wrote in.
   - Egyptian Arabic (عامية مصرية) → reply in Egyptian Arabic.
   - English → reply in English.
   - Never switch or mix languages.
2. LENGTH: 1–2 short sentences ONLY. Your reply will be read aloud while the user is driving.
3. ACCURACY: Use ONLY the trip context provided to answer navigation questions.
   Do NOT invent or guess traffic conditions, camera locations, distances, or times
   that are not explicitly in the context. If the context doesn't have the answer,
   say so naturally (e.g., "معنديش معلومات عن ده دلوقتي" / "I don't have that info right now").
4. TONE: Calm, warm, and direct — like a knowledgeable friend in the passenger seat.
5. GENERAL QUESTIONS: If the question has nothing to do with navigation,
   answer it briefly and helpfully — you are a general assistant too."""

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
    Shared chat-completion logic for BOTH /chat and /voice — identical behavior.
    Builds the same system prompt + trip-context block, calls OpenAI chat
    completions via direct httpx, and returns a ChatResponse. Never raises:
    on any failure it returns a friendly fallback ChatResponse.

    Structured for Phase 3 function-calling: add "tools"/"tool_choice" to the
    json payload below when actions are introduced.
    """
    lang = _detect_language(message)

    # ── Guard: API key ───────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY env var is not set")
        return _fallback(lang, "no_key")

    # ── Build messages ───────────────────────────────────────────────────────
    ctx_block    = _build_context_block(trip_context)
    user_content = message + ctx_block

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # ── Call OpenAI directly via httpx (no openai package needed) ────────────
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENAI_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      OPENAI_MODEL,
                    "messages":   messages,
                    "max_tokens": MAX_TOKENS,
                    # ── Phase 3 hook — add "tools": [...] here for action support ──
                },
            )
            response.raise_for_status()
            data       = response.json()
            reply_text = data["choices"][0]["message"]["content"].strip()
            tokens     = data.get("usage", {}).get("total_tokens", "?")

        if not reply_text:
            return _fallback(lang, "error")

    except httpx.TimeoutException:
        logger.warning("OpenAI request timed out")
        return _fallback(lang, "timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI HTTP error {e.response.status_code}: {e.response.text[:200]}")
        return _fallback(lang, "error")
    except Exception as e:
        logger.error(f"Unexpected assistant error: {e}")
        return _fallback(lang, "error")

    logger.info(f"assistant_chat: model={OPENAI_MODEL} lang={lang} tokens_used={tokens}")

    return ChatResponse(reply=reply_text, language=lang)


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
