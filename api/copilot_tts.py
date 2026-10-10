"""
api/copilot_tts.py — Sekka's voice: POST /api/copilot/speak {text, lang} → audio/mpeg.

The assistant used to be read aloud by the phone's own Cloud TTS call with an `ar-XA` voice: Modern Standard
Arabic phonology reading Egyptian words (ق as a hard q, ج as dʒ) — Egyptian wording in a non-Egyptian voice.
Sekka now speaks with ONE voice in both languages, steered to a Cairene accent for Arabic, from the same
provider and key the copilot already uses (no new credential to deploy).

Rules
  - Text in, MP3 out; the bytes are the provider's, never re-encoded. The phones and the cars play them exactly
    like the clips they synthesised before, and keep their own synthesis as the fallback: a failure, a cap or a
    slow answer here costs a voice, never a reply.
  - It speaks ASSISTANT text only: at most MAX_CHARS, and a per-user daily character budget
    (COPILOT_TTS_CHARS_PER_DAY) so the endpoint is not a free TTS proxy. Cache hits are never charged.
  - An in-memory LRU (MEM_MAX_BYTES) keyed by (model, voice, lang, text): the short lines every driver hears
    («تمام.», "Done.", the catalog's fallbacks) are synthesised once per instance.

Cost: gpt-4o-mini-tts ≈ $0.015 per audio minute — a 3-second reply ≈ $0.0008.
Env: OPENAI_API_KEY, COPILOT_TTS_MODEL (default gpt-4o-mini-tts), COPILOT_TTS_VOICE (default "coral").
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import httpx

logger = logging.getLogger("routemind.copilot.tts")

TTS_URL = "https://api.openai.com/v1/audio/speech"
TTS_MODEL = os.getenv("COPILOT_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("COPILOT_TTS_VOICE", "coral")
MAX_CHARS = 400
MEM_MAX_BYTES = 8 * 1024 * 1024

# The persona the voice carries. The accent notes are what an ar-XA voice could never do: the Cairene
# glottal-stop ق, the hard-g ج, Egyptian vowels — the same rules the navigation voice's phrasebook follows
# (app repo tools/masri_voice/pronounce.py).
INSTRUCTIONS = {
    "ar": ("You are Sekka (سِكّة), a warm, confident young Egyptian co-driver. Speak natural Cairene Egyptian "
           "Arabic (masri), never Modern Standard Arabic: pronounce ق as a glottal stop the way Cairenes do "
           "(الطريق → el-ṭarī’), ج as a hard g (جامعة → gamʿa), ث as s or t and ذ as z or d, with Egyptian "
           "vowels. Brisk and clear, like a friend talking to someone who is driving; calm and crisp for "
           "warnings. English names and brands stay in natural English."),
    "en": ("You are Sekka, a warm, confident co-driver from Cairo. Natural, friendly, clear English with a "
           "light Egyptian warmth. Brisk, like talking to a friend who is driving; calm and crisp for "
           "warnings. Arabic place names are pronounced the Egyptian way."),
}

_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))


class _Lru:
    def __init__(self, max_bytes: int):
        self.max = max_bytes
        self.size = 0
        self.d: "OrderedDict[str, bytes]" = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, k: str) -> Optional[bytes]:
        with self.lock:
            v = self.d.get(k)
            if v is None:
                self.misses += 1
                return None
            self.d.move_to_end(k)
            self.hits += 1
            return v

    def put(self, k: str, v: bytes) -> None:
        with self.lock:
            if k in self.d:
                return
            self.d[k] = v
            self.size += len(v)
            while self.size > self.max and self.d:
                _, old = self.d.popitem(last=False)
                self.size -= len(old)


CACHE = _Lru(MEM_MAX_BYTES)


def cache_key(text: str, lang: str) -> str:
    return hashlib.sha1(f"{TTS_MODEL}|{TTS_VOICE}|{lang}|{text}".encode("utf-8")).hexdigest()


def clean(text: str) -> str:
    """The spoken form: the Egyptian nav voice's segment joiner (U+2063) and stray markup never reach TTS."""
    t = (text or "").replace("⁣", " ").replace("*", "").replace("#", "")
    return " ".join(t.split())[:MAX_CHARS]


async def synthesize(text: str, lang: str) -> Tuple[Optional[bytes], bool]:
    """(mp3 bytes or None, from_cache). Never raises."""
    from api import copilot as base
    lang = "ar" if lang == "ar" else "en"
    text = clean(text)
    if not text:
        return None, False
    k = cache_key(text, lang)
    hit = CACHE.get(k)
    if hit is not None:
        return hit, True
    if not base.OPENAI_KEY:
        return None, False
    payload = {"model": TTS_MODEL, "voice": TTS_VOICE, "input": text, "response_format": "mp3"}
    if "tts-1" not in TTS_MODEL:                  # the gpt-4o-*-tts family takes steering instructions
        payload["instructions"] = INSTRUCTIONS[lang]
    try:
        resp = await _client.post(TTS_URL, json=payload,
                                  headers={"Authorization": f"Bearer {base.OPENAI_KEY}"})
    except Exception as e:
        logger.warning(f"copilot tts transport failed: {e}")
        return None, False
    if resp.status_code != 200 or not resp.content:
        logger.warning(f"copilot tts {resp.status_code}: {resp.text[:160]!r}")
        return None, False
    CACHE.put(k, resp.content)
    return resp.content, False


# ── Sekka's voice INSIDE the turn's stream (latency) ──────────────────────────
# The phone used to ask POST /copilot/speak for each sentence after the sentence reached it: one more round trip
# from Egypt to the backend's region (~200 ms) before the first word, and on a scaled-out service a /speak that
# lands on another instance than the turn misses this instance's cache and pays for the synthesis twice. A client
# that sends `inline_voice: true` instead gets each sentence's MP3 in the same NDJSON stream, synthesised the
# moment the sentence is written:
#     {"t":"audio","text":<the sentence as sent in its delta>,"lang":"ar"|"en","b64":<mp3 base64>|null}
# `meta` carries "inline_voice": true when the stream will do this (an older backend never says so, and the client
# keeps its /speak path). The stream closes only after every sentence's audio line. b64 null = no voice for that
# sentence (budget, provider failure): the client speaks it with its own voice at once instead of waiting.
INLINE_PARALLEL = 2                         # sentences synthesised at once (the first one first)


async def with_inline_voice(lines, limit_key=None):
    """Wrap a turn's NDJSON line stream: pass every line through, and add one `audio` line per spoken sentence."""
    import asyncio
    import base64
    import json as _json
    from api.copilot_lang import split_sentences
    from api.usage_limits import TTS_CHARS, tts_chars_per_day

    q: asyncio.Queue = asyncio.Queue()
    gate = asyncio.Semaphore(INLINE_PARALLEL)
    state = {"lang": "en", "pending": 0}

    async def pump():
        try:
            async for line in lines:
                await q.put(("line", line))
        finally:
            await q.put(("eof", None))

    async def synth(text: str, lang: str):
        b64 = None
        try:
            async with gate:
                t = clean(text)
                if t:
                    charged = CACHE.get(cache_key(t, lang)) is not None or not limit_key or \
                        TTS_CHARS.take(limit_key, tts_chars_per_day(), amount=len(t))
                    if charged:
                        audio, _ = await synthesize(t, lang)
                        if audio:
                            b64 = base64.b64encode(audio).decode("ascii")
        except Exception as e:                       # a voice problem never costs the reply
            logger.warning(f"inline voice failed: {e}")
        await q.put(("audio", {"t": "audio", "text": text, "lang": lang, "b64": b64}))

    pump_task = asyncio.ensure_future(pump())
    eof = False
    try:
        while not eof or state["pending"] > 0:
            kind, v = await q.get()
            if kind == "eof":
                eof = True
                continue
            if kind == "audio":
                state["pending"] -= 1
                yield _json.dumps(v, ensure_ascii=False) + "\n"
                continue
            try:
                obj = _json.loads(v)
            except ValueError:
                yield v
                continue
            t = obj.get("t")
            if t == "meta":
                state["lang"] = "ar" if obj.get("lang") == "ar" else "en"
                obj["inline_voice"] = True
                yield _json.dumps(obj, ensure_ascii=False) + "\n"
                continue
            yield v
            if t == "delta":
                # the client cuts the stream into sentences the same way: one clip per sentence it will speak
                parts, _ = split_sentences(obj.get("text") or "", force=True)
                for part in parts:
                    state["pending"] += 1
                    asyncio.ensure_future(synth(part, state["lang"]))
            elif t == "done" and obj.get("lang") in ("ar", "en"):
                state["lang"] = obj["lang"]
    finally:
        if not pump_task.done():
            pump_task.cancel()
