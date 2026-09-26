"""
api/copilot_stt.py — Sekka's ears: ONE language-agnostic transcription of the driver's own audio.

WHY THIS EXISTS (the language bug, closed at its root)
  The phones used to open an on-device recognizer in ONE language — the conversation's last one — BEFORE the
  driver said a word (Android `RecognizerIntent` with `EXTRA_LANGUAGE`, iOS `SFSpeechRecognizer(locale:)`).
  A monolingual recognizer does not detect language; it FORCES what it hears into its own: English spoken into
  the ar-EG recognizer came back as Arabic-script noise («هاو لونج ليفت»), Egyptian Arabic spoken into en-US
  came back as English words. Everything downstream (the resolver, the stickiness, the low-confidence guard, the
  iOS dual-recognizer race whose tie went to the sticky language because Apple reports ~0 confidence) was
  guessing the language back from text that had already lost it. No text rule can recover what the recognizer
  threw away, so the bug kept coming back in a new shape.

  The fix: the phone records the audio (with an on-device voice-activity endpoint) and sends it here. The
  transcription model is NOT told a language — it decides per utterance from the sound itself, keeps each
  language in its own script and keeps code-switching as spoken («عايز أروح الـ Ring Road»). The transcript is
  therefore honest, and the one resolver (api/copilot_lang.py) reads a text whose script IS the language spoken.

COST (a top company priority): gpt-4o-mini-transcribe is $0.003 / audio minute — a 4-second request costs
  $0.0002, about a tenth of the model turn it feeds. The phone never uploads silence (its VAD only sends once
  speech started and stops at the end-point), clips are capped at MAX_AUDIO_S, and every user has a daily audio
  budget (api/usage_limits.py `COPILOT_VOICE_SECONDS_PER_DAY`).

Env: OPENAI_API_KEY (already set for the copilot), COPILOT_STT_MODEL (default gpt-4o-mini-transcribe;
     gpt-4o-transcribe is the higher-accuracy, 2x-price option — an env flip, no code change).
"""

from __future__ import annotations

import logging
import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("routemind.copilot.stt")

STT_URL = "https://api.openai.com/v1/audio/transcriptions"
STT_MODEL = os.getenv("COPILOT_STT_MODEL", "gpt-4o-mini-transcribe")

MAX_AUDIO_BYTES = 1_600_000       # 16 kHz mono 16-bit WAV = 32 kB/s → ~50 s; the clip cap below binds first
MAX_AUDIO_S = 25.0                # the phone's recorder stops at 20 s; a little slack for headers / rounding
MIN_AUDIO_S = 0.25                # shorter than a syllable → nothing to transcribe

# Neutral between the two languages on purpose: both scripts, both directions of code-switching, the short
# answers a driver actually gives. It must NEVER be the conversation's last turn (that would re-introduce the
# sticky bias this module exists to remove) — the model's language decision has to come from the audio.
STT_PROMPT = (
    "Sekka, سِكّة. A driver in Egypt talks to the car assistant in Egyptian Arabic or in English, and often "
    "mixes both in one sentence. Arabic speech is written in Arabic script, English speech in English. "
    "عايز أروح الدائري. Take me to the Ring Road. فين أقرب بنزينة؟ Where's the nearest gas station? "
    "خدني عالـ Ring Road. أيوه، لأ، تمام، خلاص، شكرًا. Yes, no, okay, thanks."
)

_client = httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0))


@dataclass
class SttResult:
    text: str                          # the honest transcript ("" = nothing was said)
    confidence: Optional[float]        # exp(mean token log-prob) when the model reports it
    duration_s: Optional[float]        # from the WAV header when the clip is WAV
    phantom: bool = False              # the model produced one of its known silence phrases
    error: Optional[str] = None        # transport / API failure (the turn speaks err_stt)


# ── The clip ─────────────────────────────────────────────────────────────────
def wav_duration_s(data: bytes) -> Optional[float]:
    """Duration of a RIFF/WAVE clip from its header (fmt byte-rate + data size); None when it is not WAV.
    Pure — the phones send 16 kHz mono PCM WAV, and the duration is what the audio budget is charged."""
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, byte_rate, data_len = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        body = pos + 8
        if cid == b"fmt " and size >= 16 and body + 16 <= len(data):
            byte_rate = struct.unpack("<I", data[body + 8:body + 12])[0]
        elif cid == b"data":
            # a streaming writer may leave 0 / 0xFFFFFFFF in the size field — take what actually arrived
            data_len = min(size, len(data) - body) if 0 < size < 0xFFFFFFFF else len(data) - body
            break
        pos = body + size + (size & 1)
    if not byte_rate or data_len is None:
        return None
    return data_len / float(byte_rate)


# ── Phantom transcripts ──────────────────────────────────────────────────────
# Transcription models trained on subtitles "hear" these in silence / road noise. The subtitle credits are
# never a driver's words; the short courtesy phrases ARE sometimes real, so they are dropped only when the
# model itself was unsure.
_ALWAYS_PHANTOM = re.compile(
    r"(thanks? (you )?for watching|please subscribe|like and subscribe|subtitles? by|amara\.org"
    r"|اشتركوا في القناة|لا تنسوا الاشتراك|ترجمة نانسي|نانسي قنقر|شكرا للمشاهدة|شكرا على المشاهدة"
    r"|شكرا لكم على المشاهدة|\[موسيقى\]|\(موسيقى\)|\[music\]|\(music\))",
    re.IGNORECASE)
_SHORT_PHANTOMS = {"you", "thank you", "thanks", "bye", "okay", "ok", "so", "the", "uh", "um", "hmm",
                   "شكرا", "شكرًا", "مع السلامة", "السلام عليكم", "ممم", "اه"}
UNSURE_CONF = 0.5


def _norm(s: str) -> str:
    return re.sub(r"[\s.,!?؟،…\"'«»:;\-]+", " ", (s or "").lower()).strip()


def is_phantom(text: str, confidence: Optional[float]) -> bool:
    t = _norm(text)
    if not t:
        return True
    if _ALWAYS_PHANTOM.search(text or ""):
        return True
    if t in _SHORT_PHANTOMS and confidence is not None and confidence < UNSURE_CONF:
        return True
    # a single repeated token ("you you you", "ممم ممم") is the loop signature of a silent clip
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    return False


def confidence_from_logprobs(logprobs) -> Optional[float]:
    """exp(mean log-prob) over the transcript's tokens — the model's own certainty, 0..1."""
    try:
        vals = [float(x["logprob"]) for x in (logprobs or []) if x.get("logprob") is not None]
    except (TypeError, ValueError, AttributeError):
        return None
    if not vals:
        return None
    return max(0.0, min(1.0, math.exp(sum(vals) / len(vals))))


# ── The call ─────────────────────────────────────────────────────────────────
async def transcribe(audio: bytes, filename: str = "speech.wav",
                     content_type: str = "audio/wav") -> SttResult:
    """One transcription, language decided by the audio. Never raises: a failure is SttResult.error."""
    from api import copilot as base              # the one OpenAI key the copilot already uses

    dur = wav_duration_s(audio)
    if dur is not None and dur < MIN_AUDIO_S:
        return SttResult("", None, dur)
    if not base.OPENAI_KEY:
        return SttResult("", None, dur, error="not_configured")
    data = {"model": STT_MODEL, "response_format": "json", "temperature": "0", "prompt": STT_PROMPT}
    # logprobs = the model's own confidence (gpt-4o-*-transcribe only; whisper-1 rejects the field)
    if "transcribe" in STT_MODEL:
        data["include[]"] = "logprobs"
    try:
        resp = await _client.post(STT_URL, headers={"Authorization": f"Bearer {base.OPENAI_KEY}"},
                                  data=data, files={"file": (filename, audio, content_type)})
    except Exception as e:                       # network: the turn says so in the conversation's language
        logger.warning(f"copilot stt transport failed: {e}")
        return SttResult("", None, dur, error="transport")
    if resp.status_code != 200:
        logger.warning(f"copilot stt {resp.status_code}: {resp.text[:200]!r}")
        return SttResult("", None, dur, error=f"http_{resp.status_code}")
    try:
        body = resp.json()
    except ValueError:
        return SttResult("", None, dur, error="bad_json")
    text = (body.get("text") or "").strip()
    conf = confidence_from_logprobs(body.get("logprobs"))
    if is_phantom(text, conf):
        logger.info(f"copilot stt phantom dropped: {text[:40]!r} conf={conf}")
        return SttResult("", conf, dur, phantom=bool(text))
    logger.info(f"copilot stt ok: {len(text)} chars conf={conf} dur={dur}")
    return SttResult(text, conf, dur)
