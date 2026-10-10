# -*- coding: utf-8 -*-
"""
Sekka's ears — the bilingual jury's arbiter (api/copilot_stt.py decide()), on what the three jurors
realistically write for the same clip.

THE PROPERTY: the language of the turn is the language the driver SPOKE — Arabic in → Arabic, English in →
English — on long sentences, mixed speech, one-word answers and mid-conversation switches, whatever the
conversation's previous language was. A juror's failure modes are part of the fixture, not an exception:
  • the forced-English juror translating Arabic audio (fluent English, decent confidence),
  • the forced-Arabic juror transliterating English audio («هاو لونج إز ليفت») or translating it,
  • a forced juror ignoring its language and writing the other script (the audio overruled it),
  • the unprompted juror hearing an interjection («آه», "ok") that belongs to neither language,
  • a juror that errored or timed out.

Run:  python -m pytest tests/test_copilot_stt_jury.py -q
"""

import asyncio

import pytest

from api import copilot as base
from api import copilot_stt as stt
from api.copilot_lang import resolve_language
from api.copilot_stt import Juror, decide


def J(name, text, conf=0.85):
    return Juror(name, text, conf)


def E(name):
    return Juror(name, error="http_500")


# (case, ar juror, en juror, auto juror, prev_lang, expected language, expected words come from)
CASES = [
    # ── Arabic spoken ──────────────────────────────────────────────────────────
    ("ar long, en translates", J("ar", "عايز أروح الدائري من أقصر طريق", .9),
     J("en", "I want to go to the Ring Road the shortest way", .72), J("auto", "عايز أروح الدائري من أقصر طريق", .9),
     "en", "ar", "ar"),
    ("ar question, en romanizes", J("ar", "فين أقرب بنزينة؟", .88), J("en", "Fein a2rab banzeena?", .5),
     J("auto", "فين أقرب بنزينة؟", .86), "en", "ar", "ar"),
    ("ar, en juror wrote Arabic anyway", J("ar", "الزحمة دي عادية ولا لأ", .9), J("en", "الزحمة دي عادية ولا لا", .8),
     J("auto", "الزحمة دي عادية ولا لأ", .9), "en", "ar", "ar"),
    ("ar, auto missing", J("ar", "شيل الوقفة اللي جاية", .9), J("en", "Remove the next stop", .7), None,
     "en", "ar", "ar"),
    ("ar mixed with English nouns", J("ar", "خدني على رينج رود من غير كارتة", .88),
     J("en", "Take me to Ring Road without toll", .6), J("auto", "خدني على Ring Road من غير كارتة", .87),
     "en", "ar", "ar"),
    ("ar one word, conversation English", J("ar", "لأ", .9), J("en", "No.", .62), J("auto", "لا", .8),
     "en", "ar", "ar"),
    ("ar yes word تمام, conversation English", J("ar", "تمام", .92), J("en", "Tamam.", .55), J("auto", "تمام", .9),
     "en", "ar", "ar"),
    ("ar, auto errored, en gibberish", J("ar", "وريني الطريق التاني", .85), J("en", "Wareeni el tare2 el tany", .4),
     E("auto"), "en", "ar", "ar"),
    # ── English spoken ─────────────────────────────────────────────────────────
    ("en long, ar transliterates", J("ar", "هاو لونج إز ليفت تو ذا ديستنيشن", .55),
     J("en", "How long is left to the destination?", .95), J("auto", "How long is left to the destination?", .95),
     "ar", "en", "en"),
    ("en, ar translates", J("ar", "فاضل قد إيه على ما نوصل؟", .62), J("en", "How long until we get there?", .94),
     J("auto", "How long until we get there?", .93), "ar", "en", "en"),
    ("en, ar juror wrote Latin anyway", J("ar", "Take me home please", .7), J("en", "Take me home, please.", .95),
     J("auto", "Take me home, please.", .95), "ar", "en", "en"),
    ("en one word yes, conversation Arabic", J("ar", "يس", .5), J("en", "Yes.", .95), J("auto", "Yes.", .93),
     "ar", "en", "en"),
    ("en, auto missing", J("ar", "شو مي ذا ترافيك", .5), J("en", "Show me the traffic", .94), None,
     "ar", "en", "en"),
    ("en with an Arabic place name", J("ar", "تيك مي تو التجمع الخامس", .6),
     J("en", "Take me to Tagamoa El Khames", .9), J("auto", "Take me to Tagamoa El Khames", .9), "ar", "en", "en"),
    # ── neither: interjections stay with the conversation ──────────────────────
    ("interjection after Arabic", J("ar", "آه", .7), J("en", "Ah.", .66), J("auto", "Ah.", .6), "ar", "ar", "ar"),
    ("interjection after English", J("ar", "آه", .7), J("en", "Ah.", .66), J("auto", "Ah.", .6), "en", "en", "en"),
    ("okay after Arabic", J("ar", "أوكي", .8), J("en", "Okay.", .82), J("auto", "Okay.", .8), "ar", "ar", "ar"),
    ("okay after English", J("ar", "أوكي", .8), J("en", "Okay.", .82), J("auto", "Okay.", .8), "en", "en", "en"),
]


@pytest.mark.parametrize("case,ar,en,auto,prev,want,frm", CASES, ids=[c[0] for c in CASES])
def test_the_language_is_the_language_spoken(case, ar, en, auto, prev, want, frm):
    lang, chosen, why = decide(ar, en, auto, prev)
    assert lang == want, f"{case}: {why}"
    assert chosen is not None and chosen.name in (frm, "auto"), f"{case}: words from {chosen.name}"
    # the words handed to the brain are in the script of the verdict (or carry no letters)
    sc = stt._script(chosen.text)[0]
    assert sc in (want, None), f"{case}: chose {chosen.text!r}"


@pytest.mark.parametrize("prev", [None, "ar", "en"])
def test_a_clear_switch_never_depends_on_the_conversation(prev):
    ar_audio = decide(J("ar", "عايز أغير الوجهة للبيت", .9), J("en", "I want to change the destination home", .7),
                      J("auto", "عايز أغير الوجهة للبيت", .9), prev)
    en_audio = decide(J("ar", "ويتش واي إز فاستر", .5), J("en", "Which way is faster?", .95),
                      J("auto", "Which way is faster?", .95), prev)
    assert ar_audio[0] == "ar" and en_audio[0] == "en"


def test_nothing_heard():
    lang, chosen, _ = decide(Juror("ar", phantom=True), Juror("en", ""), Juror("auto", error="x"), "ar")
    assert lang is None and chosen is None


def test_the_turn_is_pinned_to_the_verdict_and_an_explicit_request_still_wins():
    # the verdict is the turn's language, whatever the previous one was …
    r = resolve_language("فين أقرب بنزينة", prev_lang="en", stt_lang="ar", stt_confidence=0.9, stt_source="audio")
    assert (r.lang, r.source) == ("ar", "audio")
    r = resolve_language("ok", prev_lang="ar", stt_lang="en", stt_confidence=0.9, stt_source="audio")
    assert r.lang == "en"
    # … except a spoken request for the other language
    r = resolve_language("كلمني انجليزي", prev_lang="ar", stt_lang="ar", stt_confidence=0.9, stt_source="audio")
    assert r.lang == "en" and r.source == "explicit"
    # no verdict (one juror): the text evidence decides, never the wrong-mic guard
    r = resolve_language("How long is left", prev_lang="ar", stt_lang=None, stt_confidence=0.3, stt_source="audio")
    assert r.lang == "en" and r.unreliable


class _R:
    def __init__(self, status, body):
        self.status_code, self._b, self.text = status, body, ""

    def json(self):
        return self._b


def test_a_slow_juror_never_holds_the_driver(monkeypatch):
    """Two jurors answered: the third gets the grace window, then the verdict is taken without it."""
    async def slow_or_fast(url, headers=None, data=None, files=None):
        if "language" not in data:
            await asyncio.sleep(5)
        if data.get("language") == "ar":
            return _R(200, {"text": "وريني الزحمة", "logprobs": [{"logprob": -0.1}]})
        return _R(200, {"text": "Show me the traffic", "logprobs": [{"logprob": -0.5}]})

    class C:
        post = staticmethod(slow_or_fast)

    monkeypatch.setattr(stt, "_client", C())
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    monkeypatch.setattr(stt, "JURY", True)
    monkeypatch.setattr(stt, "JURY_GRACE_S", 0.05)
    t0 = asyncio.run(_timed(stt.transcribe(_wav(), prev_lang="en")))
    r, secs = t0
    assert secs < 1.0 and r.lang == "ar" and r.text == "وريني الزحمة"


def test_one_juror_failing_still_decides_and_all_failing_is_an_error(monkeypatch):
    async def post(url, headers=None, data=None, files=None):
        if data.get("language") == "en":
            return _R(200, {"text": "Where am I parked?", "logprobs": [{"logprob": -0.05}]})
        return _R(500, {})

    class C:
        pass
    C.post = staticmethod(post)
    monkeypatch.setattr(stt, "_client", C())
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    monkeypatch.setattr(stt, "JURY", True)
    r = asyncio.run(stt.transcribe(_wav()))
    assert r.lang == "en" and r.text == "Where am I parked?" and r.error is None

    async def down(url, headers=None, data=None, files=None):
        raise OSError("network")
    C.post = staticmethod(down)
    r = asyncio.run(stt.transcribe(_wav()))
    assert r.error == "transport" and r.text == ""


async def _timed(coro):
    import time
    t = time.monotonic()
    r = await coro
    return r, time.monotonic() - t


def _wav(seconds=1.5, rate=16000):
    import struct
    n = int(seconds * rate)
    data = b"\x00\x00" * n
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)
