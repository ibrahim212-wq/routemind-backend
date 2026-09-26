"""
api/tts_clips.py — POST /api/tts/clips : the SHARED English guidance clip cache (cost Phase 1 / B2).

The phones and cars ask for the next few English guidance lines of their route AHEAD of the trigger
(Android nav/TtsSharedClips.kt, iOS TtsClips.swift). For each line this returns the Cloud TTS clip for
EXACTLY the request the app would send itself (endpoint + language + voice + rate + MP3 + text):
  - from the cache when anyone asked for it before (memory, then the bucket), or
  - by making that request ONCE to Cloud TTS, keeping the answer, and returning it.
The bytes returned are the bytes Cloud TTS returned — never re-encoded, never produced any other way.

Guard rails
  - Only the app's two English guidance voices are served (ALLOWED). Arabic (the MasriVoice bank, the
    Arabic cloud voice) never comes here; anything else is a 400.
  - At most MAX_TEXTS lines of at most MAX_CHARS characters per request.
  - Bounded: an in-memory LRU of MEM_MAX_BYTES; the optional bucket (TTS_CLIP_BUCKET) is bounded by its
    lifecycle rule (docs/cost/phase1-report.md, Needs Khaled).
  - The bucket, or Cloud TTS, failing costs nothing but that line: it is left out of the answer and the
    app synthesises it live at its trigger, as before. Nothing on the app's speaking path waits for this.
  - Hit rate: every request logs hits / misses; GET /api/tts/clips/stats returns the running counters.
  - Abuse: the endpoint is public (like the rest of this API), so it is not a free TTS proxy:
      * only an English guidance LINE is synthesised (is_guidance): it must start like one (GUIDANCE_RX)
        and EVERY lower-case word in it must be a word of the Directions API's English instruction grammar
        or of the app's own rewrites (NAV_WORDS); capitalised words (road names) are allowed in runs of at
        most MAX_NAME_RUN; non-Latin names pass. Anything else is simply not served (the app then
        synthesises it live, as before). This stops free text; a Title-Case phrase could still pass, so
        it is a cost limiter, not authentication;
      * a client (the RIGHTMOST X-Forwarded-For hop, which Cloud Run appends and a caller cannot forge)
        gets at most PER_CLIENT_LINES_PER_MIN lines a minute;
      * an instance synthesises at most TTS_CLIP_DAILY_SYNTH_CAP new lines a day.
    Past any of them only cached clips are served. App Check and a GLOBAL cap are deploy prerequisites
    (docs/cost/phase1-report.md, Needs Khaled).

Env: GOOGLE_TTS_API_KEY (the Cloud TTS key; without it only cache hits are served),
     TTS_CLIP_BUCKET (optional GCS bucket name; without it the cache is memory-only per instance).
"""

import asyncio
import base64
import hashlib
import logging
import os
import re
import time
from collections import OrderedDict

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger("routemind.tts_clips")
router = APIRouter()

# (endpoint, languageCode, name, speakingRate): exactly what each app platform sends.
ALLOWED = {
    ("v1beta1", "en-US", "en-US-Journey-F", 1.08),   # Android phone + Android Auto
    ("v1", "en-US", "en-US-Journey-F", 1.05),        # iPhone + CarPlay
}
MAX_TEXTS = 6
MAX_CHARS = 300
MEM_MAX_BYTES = 16 * 1024 * 1024
MAX_CLIP_BYTES = 512 * 1024
TTS_TIMEOUT_S = 5.0          # well under the app's 8 s: the server never pays for an answer the app gave up on
PER_CLIENT_LINES_PER_MIN = 120

# The shape of an English guidance line: the Directions API's English voice instructions as the app
# speaks them (an optional "Heads up — " and "In <distance>, " then a maneuver / arrival phrase).
GUIDANCE_RX = re.compile(
    r"^(?:Heads up — )?(?:[Ii]n [\d.,]+ (?:meters?|kilometers?|km|miles?|feet), )?"
    r"(?:turn|continue|keep|take|at the roundabout|make a|merge|exit|enter|go|start|drive|head|bear|"
    r"stay|you have arrived|you will arrive|you are arriving|your destination|arrive|proceed|the destination|"
    r"then|slight|sharp|use the|ramp|u-turn)\b",
    re.IGNORECASE)


# Every lower-case word of an English guidance line: the Directions API's English voice-instruction
# grammar (OSRM text instructions, v5) plus the app's rewrites (SpeechText.english / preprocessNavText:
# "Heads up —", "at the roundabout, take the third exit", "Start on", "make a U-turn").
NAV_WORDS = frozenset("""
a after ahead al and arrive arrived arriving at be bear before both but by continue destination
drive east el exit exits ferry first fifth for fork fourth go have head heads immediately in into is
it keep kilometer kilometers km lane lanes left make market meter meters mile miles feet merge north
northeast northwest of on onto or permitted proceed ramp right road roundabout rotary second seventh
sharp side sixth slight south southeast southwest start stay straight take tenth the then third
to toward towards turn u up use uturn via west will you your ninth eighth end street if
""".split())
MAX_NAME_RUN = 8
# a number with its ordinal ("3rd", "26th") is one token; so is a name with an apostrophe ("Ra's")
_WORD_RX = re.compile(r"\d+(?:st|nd|rd|th)?|[A-Za-z]+(?:'[A-Za-z]+)*")


def is_guidance(text: str) -> bool:
    t = text.strip()
    if not GUIDANCE_RX.match(t):
        return False
    for clause in re.split(r"[,.;:!?—]", t):          # a name run never spans punctuation
        run = 0
        for w in _WORD_RX.findall(clause):
            if w[0].isdigit():
                run = 0
                continue
            if w[0].isupper():
                run += 1
                if run > MAX_NAME_RUN:
                    return False
                continue
            run = 0
            if w not in NAV_WORDS:
                return False
    return True


def _daily_cap() -> int:
    try:
        return int(os.getenv("TTS_CLIP_DAILY_SYNTH_CAP", "20000"))
    except ValueError:
        return 20000


_client_window: dict = {}                 # client → (window start, lines)
_day = {"day": time.strftime("%Y-%m-%d"), "synth": 0}


def _client_allows(client: str, n: int) -> bool:
    now = time.time()
    start, used = _client_window.get(client, (now, 0))
    if now - start >= 60:
        start, used = now, 0
    if used + n > PER_CLIENT_LINES_PER_MIN:
        return False
    _client_window[client] = (start, used + n)
    if len(_client_window) > 10000:        # bounded: forget the oldest windows
        for c in sorted(_client_window, key=lambda c: _client_window[c][0])[:5000]:
            _client_window.pop(c, None)
    return True


def _synth_budget_ok() -> bool:
    today = time.strftime("%Y-%m-%d")
    if _day["day"] != today:
        _day["day"], _day["synth"] = today, 0
    return _day["synth"] < _daily_cap()


class Voice(BaseModel):
    endpoint: str
    languageCode: str
    name: str
    speakingRate: float


class ClipsRequest(BaseModel):
    voice: Voice
    texts: list[str]


def voice_tuple(v: Voice) -> tuple:
    return (v.endpoint, v.languageCode, v.name, round(float(v.speakingRate), 4))


def clip_key(vt: tuple, text: str) -> str:
    endpoint, lang, name, rate = vt
    return hashlib.sha256(f"{endpoint}|{lang}|{name}|{rate!r}|MP3|{text}".encode("utf-8")).hexdigest()


def tts_request_body(vt: tuple, text: str) -> dict:
    """The request body the app itself sends for this line (MapboxNavigationActivity.synthesizeTts,
    CloudTtsPlayer.request, MapboxNavigationViewController.synthesizeTTS, CarVoice.cloudRequest)."""
    _, lang, name, rate = vt
    return {"input": {"text": text}, "voice": {"languageCode": lang, "name": name},
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": rate}}


# ── the in-memory tier (bounded LRU) ─────────────────────────────────────────────────────────
class _MemLru:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.d: "OrderedDict[str, bytes]" = OrderedDict()

    def get(self, k: str):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k: str, v: bytes):
        if k in self.d:
            self.bytes -= len(self.d.pop(k))
        self.d[k] = v
        self.bytes += len(v)
        while self.bytes > self.max_bytes and self.d:
            _, old = self.d.popitem(last=False)
            self.bytes -= len(old)


_mem = _MemLru(MEM_MAX_BYTES)
_stats = {"requests": 0, "lines": 0, "hit_memory": 0, "hit_bucket": 0, "synthesised": 0, "failed": 0,
          "since": time.time()}
_inflight: dict = {}


# ── the optional bucket tier ─────────────────────────────────────────────────────────────────
_bucket = None
_bucket_checked = False


def _get_bucket():
    global _bucket, _bucket_checked
    if _bucket_checked:
        return _bucket
    _bucket_checked = True
    name = os.getenv("TTS_CLIP_BUCKET", "").strip()
    if not name:
        return None
    try:
        from google.cloud import storage  # firebase-admin brings google-cloud-storage
        _bucket = storage.Client().bucket(name)
    except Exception as e:  # no credentials / no library: memory-only, never an error
        logger.warning(f"tts clip bucket unavailable ({e}); memory-only")
        _bucket = None
    return _bucket


async def _bucket_get(k: str):
    b = await asyncio.to_thread(_get_bucket)   # the first call builds the client: off the event loop
    if b is None:
        return None
    try:
        return await asyncio.to_thread(lambda: b.blob(f"clips/{k}.mp3").download_as_bytes())
    except Exception:
        return None


async def _bucket_put(k: str, data: bytes):
    b = await asyncio.to_thread(_get_bucket)
    if b is None:
        return
    try:
        await asyncio.to_thread(lambda: b.blob(f"clips/{k}.mp3").upload_from_string(data, content_type="audio/mpeg"))
    except Exception as e:
        logger.warning(f"tts clip bucket write failed: {e}")


# ── Cloud TTS ────────────────────────────────────────────────────────────────────────────────
async def synthesise(vt: tuple, text: str) -> bytes | None:
    """ONE Cloud TTS request, exactly the app's. None on any failure. Replaced in tests."""
    key = os.getenv("GOOGLE_TTS_API_KEY", "").strip()
    if not key:
        return None
    url = f"https://texttospeech.googleapis.com/{vt[0]}/text:synthesize?key={key}"
    try:
        async with httpx.AsyncClient(timeout=TTS_TIMEOUT_S) as client:
            r = await client.post(url, json=tts_request_body(vt, text))
        if r.status_code != 200:
            logger.warning(f"tts synth {r.status_code}")
            return None
        return base64.b64decode(r.json()["audioContent"])
    except Exception as e:
        logger.warning(f"tts synth error: {e}")
        return None


def _keep(k: str, f) -> None:
    _inflight.pop(k, None)
    if f.cancelled() or f.exception() is not None:
        return
    v = f.result()
    if v and len(v) <= MAX_CLIP_BYTES:
        _mem.put(k, v)
        try:
            asyncio.get_running_loop().create_task(_bucket_put(k, v))
        except RuntimeError:
            pass


async def _clip(vt: tuple, text: str, may_synthesise: bool = True) -> tuple[bytes | None, str]:
    k = clip_key(vt, text)
    v = _mem.get(k)
    if v is not None:
        return v, "memory"
    v = await _bucket_get(k)
    if v:
        _mem.put(k, v)
        return v, "bucket"
    # One synthesis per key even when several devices ask at once.
    fut = _inflight.get(k)
    if fut is None:
        if not may_synthesise or not is_guidance(text) or not _synth_budget_ok():
            return None, "failed"
        _day["synth"] += 1
        fut = asyncio.ensure_future(synthesise(vt, text))
        _inflight[k] = fut
        # Keep the answer when it lands, whoever is still waiting: a cancelled first requester must not
        # lose a clip already paid for.
        fut.add_done_callback(lambda f, k=k: _keep(k, f))
        v = await asyncio.shield(fut)
        return (v, "synthesised") if v and len(v) <= MAX_CLIP_BYTES else (None, "failed")
    v = await asyncio.shield(fut)
    return (v, "synthesised") if v and len(v) <= MAX_CLIP_BYTES else (None, "failed")


@router.post("/tts/clips")
async def tts_clips(req: ClipsRequest, request: Request):
    vt = voice_tuple(req.voice)
    if vt not in ALLOWED:
        raise HTTPException(status_code=400, detail="voice not served")
    texts = []
    for t in req.texts[:MAX_TEXTS]:
        if isinstance(t, str) and t.strip() and len(t) <= MAX_CHARS and t not in texts:
            texts.append(t)
    # The RIGHTMOST X-Forwarded-For hop is the one Cloud Run appended (the caller's real address); every
    # hop left of it is whatever the caller chose to send.
    hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    client = hops[-1] if hops else (request.client.host if request.client else "?")
    allowed = _client_allows(client, len(texts))
    results = await asyncio.gather(*[_clip(vt, t, may_synthesise=allowed) for t in texts])
    clips, counts = [], {"memory": 0, "bucket": 0, "synthesised": 0, "failed": 0}
    for t, (v, src) in zip(texts, results):
        counts[src] += 1
        if v:
            clips.append({"text": t, "audio": base64.b64encode(v).decode("ascii")})
    _stats["requests"] += 1
    _stats["lines"] += len(texts)
    _stats["hit_memory"] += counts["memory"]
    _stats["hit_bucket"] += counts["bucket"]
    _stats["synthesised"] += counts["synthesised"]
    _stats["failed"] += counts["failed"]
    served = _stats["hit_memory"] + _stats["hit_bucket"]
    logger.info(f"tts clips voice={vt[0]}/{vt[3]} lines={len(texts)} hits={counts['memory'] + counts['bucket']} "
                f"synth={counts['synthesised']} failed={counts['failed']} | running hit "
                f"{served}/{max(1, _stats['lines'])} = {100 * served / max(1, _stats['lines']):.1f}%")
    return {"clips": clips, "hits": counts["memory"] + counts["bucket"], "synthesised": counts["synthesised"]}


@router.get("/tts/clips/stats")
async def tts_clips_stats():
    served = _stats["hit_memory"] + _stats["hit_bucket"]
    return dict(_stats, hit_rate=round(served / max(1, _stats["lines"]), 4), memory_bytes=_mem.bytes,
                memory_clips=len(_mem.d), bucket=bool(_get_bucket()))
