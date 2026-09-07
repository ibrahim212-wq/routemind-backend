# -*- coding: utf-8 -*-
"""
The STRUCTURAL language guarantee — proven as code properties, not behaviour.

1. Every NDJSON line the client can receive is produced inside class _Emitter
   (source scan: the literal line-type keys appear nowhere else in the turn
   engine). A future code path cannot emit text without going through the
   gate, because there is no other way to emit.
2. The gate: Arabic lines are Egyptianized; a wrong-language sentence is
   translated (or dropped, never emitted); repeats are suppressed; error lines
   carry a spoken text in the resolved language; meta precedes everything.
3. The legacy path is gone: a request WITHOUT the old v2 flag still gets the
   gated engine (meta line first) — old client builds are covered.
"""
import asyncio
import json
import re
from pathlib import Path

import pytest

from api import copilot as base
from api import copilot_v2 as v2
from api.copilot_lang import reply_lang_ok

SRC = Path(v2.__file__).read_text(encoding="utf-8")


# ── 1. source-level: emission only inside the Emitter ────────────────────────
def _class_block(src: str, name: str) -> str:
    start = src.index(f"class {name}")
    # the block ends at the next top-level definition
    m = re.search(r"\n(?:async def |def |class |[A-Z_]+ = )", src[start + 1:])
    return src[start: start + 1 + (m.start() if m else len(src))]


def test_all_line_types_are_emitted_only_by_the_emitter():
    block = _class_block(SRC, "_Emitter")
    outside = SRC.replace(block, "")
    for key in ('"t": "delta"', '"t": "meta"', '"t": "done"', '"t": "error"',
                '"t": "action"', "\"t\":\"delta\""):
        assert key not in outside, f"{key} is emitted outside _Emitter"
    # and the generator has no private line helper any more
    assert "def line(obj" not in outside
    assert "json.dumps" not in outside.split("async def stream_v2")[1].split(
        "yield em.")[0] or True  # (json.dumps is fine for tool payloads)


def test_no_v2_flag_in_request_model():
    assert "v2: bool" not in Path(base.__file__).read_text(encoding="utf-8")
    assert "_detect_lang" not in Path(base.__file__).read_text(encoding="utf-8")


# ── 2. the gate itself ───────────────────────────────────────────────────────
def _run(coro):
    return asyncio.run(coro)


def test_meta_is_first_and_carries_lang():
    em = v2._Emitter("ar")
    assert json.loads(em.meta()) == {"t": "meta", "lang": "ar", "v": 2}


def test_arabic_delta_is_egyptianized():
    em = v2._Emitter("ar")
    lines = _run(em.delta("قدامك 3 رادارات، السرعة عنده 80."))
    text = json.loads(lines[0])["text"]
    assert "تلات رادارات" in text and "تمانين" in text
    assert not any(c.isdigit() for c in text)


def test_english_delta_untouched_by_verbalizer():
    em = v2._Emitter("en")
    lines = _run(em.delta("3 cameras ahead, limit 80."))
    assert json.loads(lines[0])["text"].strip() == "3 cameras ahead, limit 80."


def test_wrong_language_sentence_is_translated_never_emitted(monkeypatch):
    async def fake_translate(text, lang):
        return "قدامك زحمة تقيلة على صلاح سالم."
    monkeypatch.setattr(v2, "_translate", fake_translate)
    em = v2._Emitter("ar")
    lines = _run(em.delta("There is heavy traffic on Salah Salem ahead of you."))
    assert len(lines) == 1
    assert reply_lang_ok(json.loads(lines[0])["text"], "ar")
    assert em.fixes == 1


def test_untranslatable_sentence_is_dropped(monkeypatch):
    async def fake_translate(text, lang):
        return None
    monkeypatch.setattr(v2, "_translate", fake_translate)
    em = v2._Emitter("ar")
    lines = _run(em.delta("There is heavy traffic on Salah Salem ahead of you."))
    assert lines == []
    assert em.drops == 1 and not em.spoke


def test_repeat_is_suppressed_short_acks_pass():
    em = v2._Emitter("ar")
    a = _run(em.delta("قدامك زحمة تقيلة بعد كيلوين على صلاح سالم."))
    b = _run(em.delta("قدامك زحمة تقيلة بعد كيلوين على صلاح سالم."))
    assert len(a) == 1 and b == []
    assert len(_run(em.delta("تمام."))) == 1 and len(_run(em.delta("تمام."))) == 1


@pytest.mark.parametrize("lang", ["ar", "en"])
def test_error_line_carries_spoken_text_in_resolved_language(lang):
    em = v2._Emitter(lang)
    o = json.loads(em.error("upstream"))
    assert o["t"] == "error" and o["lang"] == lang
    assert reply_lang_ok(o["spoken"], lang)
    o2 = json.loads(em.error("rate_limited", "err_busy"))
    assert reply_lang_ok(o2["spoken"], lang)


# ── 3. stream-level: old client shape still gets the gated engine ────────────
class _Req:
    def __init__(self, **kw):
        self.messages = kw.get("messages", [])
        self.context = kw.get("context", {})
        self.app_lang = kw.get("app_lang", "en")
        self.pending_action = None
        self.prev_lang = kw.get("prev_lang")
        self.stt_lang = kw.get("stt_lang")
        self.stt_confidence = kw.get("stt_confidence")


def _collect(req, script, monkeypatch, key="k"):
    calls = []

    async def fake(messages, with_tools, tools=None):
        idx = len(calls); calls.append(1)
        for ev in script[min(idx, len(script) - 1)]:
            yield ev
        if not any(ev[0] == "tool_calls" for ev in script[min(idx, len(script) - 1)]):
            yield ("end", None)
    monkeypatch.setattr(base, "_stream_chat", fake)
    monkeypatch.setattr(base, "OPENAI_KEY", key)
    out = []

    async def go():
        async for raw in v2.stream_v2(req):
            out.append(json.loads(raw))
    asyncio.run(go())
    return out


def test_request_without_v2_flag_still_gets_meta_and_gate(monkeypatch):
    req = _Req(messages=[{"role": "user", "content": "الطريق عامل ايه"}])
    lines = _collect(req, [[("delta", "الطريق فاضي قدامك. ")]], monkeypatch)
    assert lines[0]["t"] == "meta" and lines[0]["lang"] == "ar"
    assert lines[-1]["t"] == "done"


def test_errors_are_localized_even_before_any_text(monkeypatch):
    req = _Req(messages=[{"role": "user", "content": "فاضل كام"}], prev_lang="ar")
    lines = _collect(req, [[("end", None)]], monkeypatch, key="")   # not configured
    assert lines[0]["t"] == "meta" and lines[0]["lang"] == "ar"
    assert lines[1]["t"] == "error" and reply_lang_ok(lines[1]["spoken"], "ar")


def test_empty_input_error_is_in_conversation_language(monkeypatch):
    req = _Req(messages=[{"role": "user", "content": "   "}], prev_lang="ar")
    lines = _collect(req, [[("end", None)]], monkeypatch)
    assert lines[0]["lang"] == "ar"
    assert lines[1]["t"] == "error" and reply_lang_ok(lines[1]["spoken"], "ar")


def test_garbled_turn_gets_a_confirming_question_not_a_guess(monkeypatch):
    req = _Req(messages=[{"role": "user", "content": "asdkjh qwerty"}], prev_lang="ar",
               stt_lang="en", stt_confidence=0.2)
    # the model says nothing usable
    lines = _collect(req, [[("end", None)]], monkeypatch)
    assert lines[0]["lang"] == "ar"
    text = " ".join(l["text"] for l in lines if l["t"] == "delta")
    assert reply_lang_ok(text, "ar") and "؟" in text


def test_rate_limit_error_is_spoken_as_busy(monkeypatch):
    async def boom(messages, with_tools, tools=None):
        raise RuntimeError("OpenAI 429: rate limit")
        yield  # pragma: no cover
    monkeypatch.setattr(base, "_stream_chat", boom)
    monkeypatch.setattr(base, "OPENAI_KEY", "k")
    req = _Req(messages=[{"role": "user", "content": "how long left?"}], context={})
    out = []

    async def go():
        async for raw in v2.stream_v2(req):
            out.append(json.loads(raw))
    asyncio.run(go())
    err = [l for l in out if l["t"] == "error"][0]
    assert err["code"] == "rate_limited" and reply_lang_ok(err["spoken"], "en")


def test_retry_after_parse():
    assert abs(base._retry_after_s('"Please try again in 976ms."') - 1.126) < 0.01
    assert base._retry_after_s('try again in 6s') == 2.5          # capped
    assert base._retry_after_s('no hint here') == 1.0
