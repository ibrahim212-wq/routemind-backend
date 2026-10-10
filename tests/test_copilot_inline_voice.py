# -*- coding: utf-8 -*-
"""
Sekka's voice inside the turn's stream (api/copilot_tts.py with_inline_voice): every spoken sentence gets its
MP3 in the same NDJSON stream, the meta line says so, nothing else changes, the stream ends only after the last
clip, a failed synthesis is an explicit null (the client speaks it itself at once), and a budget-exhausted user
gets nulls without a provider call.

Run:  python -m pytest tests/test_copilot_inline_voice.py -q
"""

import asyncio
import base64
import json

from api import copilot as base
from api import copilot_tts as tts
from api import usage_limits as ul


def _lines(*objs):
    async def gen():
        for o in objs:
            await asyncio.sleep(0)
            yield json.dumps(o, ensure_ascii=False) + "\n"
    return gen()


def _run(gen):
    async def go():
        return [json.loads(l) async for l in gen]
    return asyncio.run(go())


def test_each_sentence_gets_its_clip_and_the_rest_passes_through(monkeypatch):
    said = []

    async def fake_synth(text, lang):
        said.append((text, lang))
        await asyncio.sleep(0.01)
        return b"MP3:" + text.encode(), False
    monkeypatch.setattr(tts, "synthesize", fake_synth)
    src = _lines({"t": "meta", "lang": "ar", "v": 2},
                 {"t": "heard", "text": "فين أقرب بنزينة", "lang": "ar"},
                 {"t": "delta", "text": "فيه بنزينة بعد كيلو. "},
                 {"t": "action", "action": {"type": "show_pois"}},
                 {"t": "delta", "text": "أضيفها؟ "},
                 {"t": "done", "expects_reply": True, "open_mic": True, "end": False, "lang": "ar"})
    out = _run(tts.with_inline_voice(src, None))
    kinds = [o["t"] for o in out]
    assert out[0]["t"] == "meta" and out[0]["inline_voice"] is True
    assert kinds.count("audio") == 2 and kinds[-1] == "audio"          # the stream ends after the last clip
    assert [o["t"] for o in out if o["t"] != "audio"] == ["meta", "heard", "delta", "action", "delta", "done"]
    clips = {o["text"]: o for o in out if o["t"] == "audio"}
    assert set(clips) == {"فيه بنزينة بعد كيلو.", "أضيفها؟"}
    assert all(c["lang"] == "ar" for c in clips.values())
    assert base64.b64decode(clips["أضيفها؟"]["b64"]) == "MP3:أضيفها؟".encode()
    assert sorted(t for t, _ in said) == sorted(clips)


def test_a_delta_with_two_sentences_is_two_clips_like_the_client_cuts_it(monkeypatch):
    async def fake_synth(text, lang):
        return b"x" * 300, False
    monkeypatch.setattr(tts, "synthesize", fake_synth)
    out = _run(tts.with_inline_voice(_lines({"t": "meta", "lang": "en"},
                                            {"t": "delta", "text": "Done. It's on the map. "},
                                            {"t": "done", "lang": "en"}), None))
    assert sorted(o["text"] for o in out if o["t"] == "audio") == ["Done.", "It's on the map."]


def test_a_failed_voice_is_an_explicit_null_and_the_budget_is_respected(monkeypatch):
    calls = []

    async def failing(text, lang):
        calls.append(text)
        return None, False
    monkeypatch.setattr(tts, "synthesize", failing)
    out = _run(tts.with_inline_voice(_lines({"t": "meta", "lang": "en"}, {"t": "delta", "text": "Okay. "},
                                            {"t": "done", "lang": "en"}), None))
    assert [o for o in out if o["t"] == "audio"][0]["b64"] is None and calls == ["Okay."]

    calls.clear()
    monkeypatch.setattr(ul, "tts_chars_per_day", lambda: 3)
    ul.TTS_CHARS.__init__()
    ul.TTS_CHARS.take("dev-1", 3, amount=5)          # the day's characters are already spoken
    out = _run(tts.with_inline_voice(_lines({"t": "meta", "lang": "en"}, {"t": "delta", "text": "A long sentence. "},
                                            {"t": "done", "lang": "en"}), "dev-1"))
    assert [o for o in out if o["t"] == "audio"][0]["b64"] is None and calls == []


def test_caps_announce_it():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(base.router, prefix="/api")
    old = base.OPENAI_KEY
    base.OPENAI_KEY = "sk-test"
    try:
        caps = TestClient(app).get("/api/copilot/caps").json()
    finally:
        base.OPENAI_KEY = old
    assert caps["inline_voice"] is True and caps["voice"] is True
