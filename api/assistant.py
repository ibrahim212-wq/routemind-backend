"""
api/assistant.py
POST /api/assistant/chat

Phase 1 — read-only AI voice assistant foundation.
TEXT IN, TEXT OUT. No voice, no trip modifications yet.

Required env var (Cloud Run secret):
  OPENAI_API_KEY   — OpenAI secret key. Set via:
                     gcloud run services update routemind-backend \\
                       --update-secrets OPENAI_API_KEY=openai-key:latest \\
                       --region us-central1

Optional env var:
  ASSISTANT_MODEL  — OpenAI model to use. Default: "gpt-4o-mini".
                     Switch to "gpt-4o" for higher quality when needed.
"""

import os
import logging
from typing import Optional, List

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("routemind.assistant")
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_MODEL = os.environ.get("ASSISTANT_MODEL", "gpt-4o-mini")
MAX_TOKENS   = 150   # short spoken replies; bump for richer responses if needed

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

# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/assistant/chat", response_model=ChatResponse)
async def assistant_chat(body: ChatRequest):
    """
    Phase 1 AI assistant — read-only, text-in / text-out.

    Accepts a free-form user message (Arabic or English) plus an optional
    TripContext object from the navigation app. Calls OpenAI and returns a
    short spoken reply in the same language the user wrote in.

    The OpenAI call is structured for Phase 3 function-calling: just
    uncomment the `tools` / `tool_choice` lines and define _ACTION_TOOLS.
    """
    lang = _detect_language(body.message)

    # ── Guard: API key ───────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY env var is not set")
        return _fallback(lang, "no_key")

    # ── Build messages ───────────────────────────────────────────────────────
    ctx_block    = _build_context_block(body.trip_context)
    user_content = body.message + ctx_block

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # ── Call OpenAI directly via httpx (no openai package needed) ────────────
    # Direct HTTP avoids the openai SDK's default client, which on Cloud Run
    # picks up ambient proxy env vars and causes connection errors.
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
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
