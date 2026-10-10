"""
api/copilot_stt.py — Sekka's ears: the driver's own audio → the words AND the language they were spoken in.

WHY THIS FILE DECIDES THE LANGUAGE (the bug, closed at its root — 2026-10-09)
  1. The phones used to open an on-device recognizer in ONE language (the conversation's last) before the
     driver spoke: a monolingual recognizer forces what it hears into its own language, so English came back
     as Arabic-script noise and Egyptian Arabic as English words. The audio path (phone records → this module)
     replaced it — but only when a 4-second capability probe at navigation start succeeded, and a cold Cloud
     Run instance (the model loads at startup) answers slower than that. One failed probe put the whole trip
     back on the monolingual recognizer. The phones now never gate the ears on that probe (app repo).
  2. On the audio path the transcriber was given ONE mixed prompt led by English. OpenAI's own guidance is
     that the prompt must match the audio's language; gpt-4o-*-transcribe treats the prompt as the preceding
     transcript, and an English-led prompt pulls an Arabic clip toward English output (and a short English
     clip toward the Arabic examples). The model also returns no language, so a wrong-script transcript was
     indistinguishable from a real switch — the resolver could only read the script it was handed.

  THE EARS NOW (a bilingual jury, one round trip):
     three transcriptions of the SAME clip run in parallel —
       • AR: language="ar" with an Egyptian-Arabic prompt (matches Arabic audio, per the guidance),
       • EN: language="en" with an English prompt,
       • AUTO: no language, NO prompt (the model's own language identification, unbiased);
     and a pure, unit-tested arbiter (decide()) reads what each juror did with the sound:
       • a forced juror that still wrote the OTHER script (AR returned Latin English / EN returned Arabic) is
         the strongest evidence there is — the audio overruled the instruction;
       • AUTO's script, from its own language identification;
       • each forced transcript checked as text: AR that is transliterated English («هاو لونج») votes English,
         EN that is Arabizi («3ayez aro7») votes Arabic, EN that is not English words votes Arabic;
       • the jurors' own acoustic confidence (mean token log-prob), compared;
     a tie keeps the conversation's language. The transcript handed to the brain is the juror of the winning
     language (decoded with the matching prompt), so the words and the language agree.
     The result carries `lang` — stream_voice pins the turn to it (an explicit "speak English" still wins).

COST (a top company priority): gpt-4o-mini-transcribe is $0.003 / audio minute. Three jurors on a 4-second
  request = 12 audio-seconds = $0.0006 — still about a quarter of the model turn it feeds, and the phone never
  uploads silence (its VAD sends only once speech started and stops at the end-point); clips are capped at
  MAX_AUDIO_S and every user has a daily audio budget (api/usage_limits.py `COPILOT_VOICE_SECONDS_PER_DAY`,
  charged once per clip). COPILOT_STT_JURY=0 runs the AUTO juror alone (one third of the cost, the old risk).

LATENCY: the jurors run in parallel; once two have answered the third gets JURY_GRACE_S more, then the verdict
  is taken without it — a straggler never holds the driver.

Env: OPENAI_API_KEY (already set for the copilot), COPILOT_STT_MODEL (default gpt-4o-mini-transcribe;
     gpt-4o-transcribe is the higher-accuracy, 2x-price option — an env flip, no code change),
     COPILOT_STT_JURY (default 1).
"""

from __future__ import annotations

import asyncio
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

# Each forced juror's prompt is written in ITS OWN language (OpenAI: "the prompt should match the audio
# language"; the model reads it as the transcript that came before). Constants — never the conversation's last
# turn, which would carry the previous language into this clip. The AUTO juror gets no prompt at all.
AR_PROMPT = ("سكة، عايز أروح الدائري. فين أقرب بنزينة؟ خدني على التجمع الخامس. لو سمحت شوف الزحمة. "
             "أيوه، لأ، تمام، خلاص، شكرًا.")
EN_PROMPT = ("Sekka, take me to the Ring Road. Where's the nearest gas station? How long is left? "
             "Yes, no, okay, thanks.")

JURY = os.getenv("COPILOT_STT_JURY", "1") not in ("0", "false", "no")
JURY_GRACE_S = 0.6              # once two jurors answered, the third gets this much longer
JURY_TIMEOUT_S = 9.0            # the whole jury, worst case

_client = httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0))


@dataclass
class SttResult:
    text: str                          # the honest transcript ("" = nothing was said)
    confidence: Optional[float]        # exp(mean token log-prob) when the model reports it
    duration_s: Optional[float]        # from the WAV header when the clip is WAV
    phantom: bool = False              # the model produced one of its known silence phrases
    error: Optional[str] = None        # transport / API failure (the turn speaks err_stt)
    lang: Optional[str] = None         # "ar" | "en" — the language the AUDIO was spoken in (None: undecided)
    lang_why: str = ""                 # the arbiter's reasons (logged; never shown)


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


# ── One juror ────────────────────────────────────────────────────────────────
@dataclass
class Juror:
    name: str                          # "ar" | "en" | "auto"
    text: str = ""
    confidence: Optional[float] = None
    error: Optional[str] = None
    phantom: bool = False

    @property
    def ok(self) -> bool:
        return not self.error and not self.phantom and bool(self.text.strip())


async def _juror(name: str, audio: bytes, filename: str, content_type: str, key: str) -> Juror:
    data = {"model": STT_MODEL, "response_format": "json", "temperature": "0"}
    if name == "ar":
        data.update(language="ar", prompt=AR_PROMPT)
    elif name == "en":
        data.update(language="en", prompt=EN_PROMPT)
    # logprobs = the model's own confidence (gpt-4o-*-transcribe only; whisper-1 rejects the field)
    if "transcribe" in STT_MODEL:
        data["include[]"] = "logprobs"
    try:
        resp = await _client.post(STT_URL, headers={"Authorization": f"Bearer {key}"},
                                  data=data, files={"file": (filename, audio, content_type)})
    except Exception as e:
        logger.warning(f"copilot stt[{name}] transport failed: {e}")
        return Juror(name, error="transport")
    if resp.status_code != 200:
        logger.warning(f"copilot stt[{name}] {resp.status_code}: {resp.text[:200]!r}")
        return Juror(name, error=f"http_{resp.status_code}")
    try:
        body = resp.json()
    except ValueError:
        return Juror(name, error="bad_json")
    text = (body.get("text") or "").strip()
    conf = confidence_from_logprobs(body.get("logprobs"))
    if is_phantom(text, conf):
        return Juror(name, "", conf, phantom=bool(text))
    return Juror(name, text, conf)


# ── The arbiter (pure — tests/test_copilot_stt_jury.py) ───────────────────────
SHORT_SWITCH_MARGIN = 2.5
_NEUTRAL = {"ah", "aah", "uh", "um", "hmm", "mm", "oh", "eh", "ok", "okay", "okey", "ه", "اه", "آه", "اها",
            "اوكي", "أوكي", "اوك", "ممم", "امم", "هه"}


def _neutral(text: str) -> bool:
    words = re.findall(r"[^\s.,!?؟،…\"'«»:;\-]+", (text or "").lower())
    return bool(words) and all(w in _NEUTRAL for w in words)


def _script(text: str):
    """("ar" | "en" | None, arabic letters, latin letters) — the script a juror actually wrote."""
    from api.copilot_lang import script_counts
    ar, en = script_counts(text or "")
    if ar + en == 0:
        return None, 0, 0
    if ar >= 0.7 * (ar + en):
        return "ar", ar, en
    if en >= 0.7 * (ar + en):
        return "en", ar, en
    return None, ar, en


def decide(ar: Juror, en: Juror, auto: Optional[Juror], prev_lang: Optional[str] = None):
    """(lang, chosen juror, reasons). lang is None only when no juror heard anything."""
    from api.copilot_lang import english_evidence, is_arabizi, is_transliterated_english
    jurors = [j for j in (ar, en, auto) if j is not None]
    if not any(j.ok for j in jurors):
        return None, None, ["nothing heard"]
    v = {"ar": 0.0, "en": 0.0}
    why = []

    # 1. a forced juror that wrote the OTHER script: the audio overruled the instruction
    if ar.ok:
        sc, a_n, l_n = _script(ar.text)
        if sc == "en" and l_n >= 3:
            v["en"] += 3.0; why.append("ar-juror wrote Latin")
        elif sc == "ar" and is_transliterated_english(ar.text):
            v["en"] += 1.5; why.append("ar-juror transliterated English")
    if en.ok:
        sc, a_n, l_n = _script(en.text)
        if sc == "ar" and a_n >= 3:
            v["ar"] += 3.0; why.append("en-juror wrote Arabic")
        elif sc == "en":
            if is_arabizi(en.text):
                v["ar"] += 1.5; why.append("en-juror wrote Arabizi")
            else:
                real, n = english_evidence(en.text)
                if n >= 3 and real / n < 0.4:
                    v["ar"] += 1.0; why.append("en-juror not English words")

    # 2. the unprompted juror's own language identification (an interjection «آه» / "ah" / "ok" belongs to
    #    neither language and casts no vote)
    if auto is not None and auto.ok and not _neutral(auto.text):
        sc, _, _ = _script(auto.text)
        if sc == "en" and is_arabizi(auto.text):
            sc = "ar"
        elif sc == "ar" and is_transliterated_english(auto.text):
            sc = "en"
        if sc:
            w = 2.0 if (auto.confidence is None or auto.confidence >= 0.5) else 1.0
            v[sc] += w; why.append(f"auto heard {sc}")

    # 3. the acoustic confidence of the two forced readings of the same sound
    if ar.ok and en.ok and ar.confidence is not None and en.confidence is not None:
        d = ar.confidence - en.confidence
        if abs(d) >= 0.04:
            side = "ar" if d > 0 else "en"
            v[side] += 1.0 + (1.0 if abs(d) >= 0.15 else 0.0) + (1.0 if abs(d) >= 0.3 else 0.0)
            why.append(f"conf {side} by {abs(d):.2f}")
    elif ar.ok != en.ok:
        side = "ar" if ar.ok else "en"
        v[side] += 0.5; why.append(f"only {side} juror answered")

    # 4. the verdict; a tie keeps the conversation's language (then the unprompted juror, then Arabic).
    #    A one- or two-word clip («آه», "yes", «تمام») carries too little sound to overturn the conversation
    #    on a slim margin: it switches only on strong evidence (a juror overruled, or two signals agree).
    short = max((len(j.text.split()) for j in jurors if j.ok), default=0) <= 2
    margin = SHORT_SWITCH_MARGIN if (short and prev_lang in ("ar", "en")) else 0.5
    lead = "ar" if v["ar"] > v["en"] else "en"
    if abs(v["ar"] - v["en"]) >= margin or (abs(v["ar"] - v["en"]) >= 0.5 and lead == prev_lang):
        lang = "ar" if v["ar"] > v["en"] else "en"
    else:
        auto_sc = _script(auto.text)[0] if (auto is not None and auto.ok) else None
        lang = prev_lang if prev_lang in ("ar", "en") else (auto_sc or ("ar" if ar.ok else "en"))
        why.append(f"tie → {lang}")

    # 5. the words: the juror of the winning language when it wrote that language, else whoever did
    def writes(j: Optional[Juror], lg: str) -> bool:
        return j is not None and j.ok and (_script(j.text)[0] in (lg, None))
    own = ar if lang == "ar" else en
    other = en if lang == "ar" else ar
    if writes(own, lang):
        chosen = own
    elif writes(auto, lang):
        chosen = auto
    elif writes(other, lang):
        chosen = other
    else:
        chosen = next(j for j in (own, auto, other) if j is not None and j.ok)
    why.append(f"v={v['ar']:.1f}/{v['en']:.1f}")
    return lang, chosen, why


# ── The call ─────────────────────────────────────────────────────────────────
async def transcribe(audio: bytes, filename: str = "speech.wav",
                     content_type: str = "audio/wav", prev_lang: Optional[str] = None) -> SttResult:
    """The words and their language, decided by the audio. Never raises: a failure is SttResult.error."""
    from api import copilot as base              # the one OpenAI key the copilot already uses

    dur = wav_duration_s(audio)
    if dur is not None and dur < MIN_AUDIO_S:
        return SttResult("", None, dur)
    if not base.OPENAI_KEY:
        return SttResult("", None, dur, error="not_configured")
    key = base.OPENAI_KEY
    if not JURY:
        j = await _juror("auto", audio, filename, content_type, key)
        if j.error:
            return SttResult("", None, dur, error=j.error)
        if not j.ok:
            return SttResult("", j.confidence, dur, phantom=j.phantom)
        return SttResult(j.text, j.confidence, dur)

    names = ("ar", "en", "auto")
    tasks = {asyncio.ensure_future(_juror(n, audio, filename, content_type, key)): n for n in names}
    got = {}
    pending = set(tasks)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + JURY_TIMEOUT_S

    def collect(done):
        for t in done:
            got[tasks[t]] = t.result()

    while pending and len(got) < 2:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        done, pending = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            break
        collect(done)
    if pending and len(got) >= 2:               # the grace window starts once two have answered
        done, pending = await asyncio.wait(pending, timeout=min(JURY_GRACE_S, max(0.0, deadline - loop.time())))
        collect(done)
    for t in pending:
        t.cancel()
    ar = got.get("ar") or Juror("ar", error="late")
    en = got.get("en") or Juror("en", error="late")
    auto = got.get("auto")
    if all(j.error for j in (ar, en, auto or Juror("auto", error="late"))):
        return SttResult("", None, dur, error=ar.error or en.error or "transport")
    lang, chosen, why = decide(ar, en, auto, prev_lang)
    if chosen is None:
        phantom = any(j.phantom for j in (ar, en, auto) if j is not None)
        conf = next((j.confidence for j in (auto, ar, en) if j is not None and j.confidence is not None), None)
        logger.info(f"copilot stt: nothing heard (phantom={phantom})")
        return SttResult("", conf, dur, phantom=phantom, lang_why="; ".join(why))
    logger.info(f"copilot stt jury: lang={lang} from={chosen.name} conf={chosen.confidence} "
                f"dur={dur} why={'; '.join(why)}")
    return SttResult(chosen.text, chosen.confidence, dur, lang=lang, lang_why="; ".join(why))
