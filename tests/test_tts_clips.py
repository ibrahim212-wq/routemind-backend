"""Tests for api/tts_clips.py (cost Phase 1 / B2): the shared English guidance clip cache.

Cloud TTS is replaced by a counting fake; no network, no bucket.
    python -m pytest tests/test_tts_clips.py -q
"""
import asyncio
import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.tts_clips as tc

ANDROID = {"endpoint": "v1beta1", "languageCode": "en-US", "name": "en-US-Journey-F", "speakingRate": 1.08}
IOS = {"endpoint": "v1", "languageCode": "en-US", "name": "en-US-Journey-F", "speakingRate": 1.05}


@pytest.fixture()
def client(monkeypatch):
    calls = []

    async def fake_synth(vt, text):
        calls.append((vt, text))
        await asyncio.sleep(0.01)
        if text == "Turn left onto FAIL.":
            return None
        return ("MP3:%s:%s" % (vt[0], text)).encode("utf-8") * 20

    monkeypatch.setattr(tc, "synthesise", fake_synth)
    monkeypatch.setattr(tc, "_mem", tc._MemLru(tc.MEM_MAX_BYTES))
    monkeypatch.setattr(tc, "_client_window", {})
    monkeypatch.setattr(tc, "_day", {"day": "x", "synth": 0})
    monkeypatch.setattr(tc, "_bucket_checked", True)
    monkeypatch.setattr(tc, "_bucket", None)
    app = FastAPI()
    app.include_router(tc.router, prefix="/api")
    c = TestClient(app)
    c.calls = calls
    return c


def post(c, voice, texts, ip="1.2.3.4"):
    return c.post("/api/tts/clips", json={"voice": voice, "texts": texts}, headers={"x-forwarded-for": ip})


def test_only_the_app_english_voices(client):
    for bad in ({**ANDROID, "name": "ar-XA-Chirp3-HD-Aoede", "languageCode": "ar-XA"},
                {**ANDROID, "speakingRate": 1.0}, {**IOS, "endpoint": "v1beta1"}):
        assert post(client, bad, ["Turn left."]).status_code == 400
    assert client.calls == []


def test_bytes_are_exactly_what_tts_returned_and_a_repeat_is_a_hit(client):
    r = post(client, ANDROID, ["Turn left.", "In 400 meters, turn right."]).json()
    assert r["synthesised"] == 2 and r["hits"] == 0
    got = {c["text"]: base64.b64decode(c["audio"]) for c in r["clips"]}
    assert got["Turn left."] == b"MP3:v1beta1:Turn left." * 20
    r2 = post(client, ANDROID, ["Turn left."]).json()
    assert r2["hits"] == 1 and r2["synthesised"] == 0
    assert base64.b64decode(r2["clips"][0]["audio"]) == got["Turn left."]
    assert len(client.calls) == 2


def test_voices_never_share_a_clip(client):
    post(client, ANDROID, ["Turn left."])
    r = post(client, IOS, ["Turn left."]).json()
    assert r["synthesised"] == 1
    assert base64.b64decode(r["clips"][0]["audio"]).startswith(b"MP3:v1:")


def test_the_request_the_server_makes_is_the_apps_own():
    vt = tc.voice_tuple(tc.Voice(**IOS))
    assert tc.tts_request_body(vt, "Turn left.") == {
        "input": {"text": "Turn left."},
        "voice": {"languageCode": "en-US", "name": "en-US-Journey-F"},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.05}}


def test_a_long_float_rate_from_json_is_the_same_voice(client):
    r = post(client, {**IOS, "speakingRate": 1.0500000000000000444}, ["Turn left."])
    assert r.status_code == 200


def test_failure_leaves_the_line_out_and_is_not_cached(client):
    r = post(client, ANDROID, ["Turn left onto FAIL.", "Turn left."]).json()
    assert [c["text"] for c in r["clips"]] == ["Turn left."]
    post(client, ANDROID, ["Turn left onto FAIL."])
    assert sum(1 for _, t in client.calls if t == "Turn left onto FAIL.") == 2


def test_caps_on_count_length_and_duplicates(client):
    texts = ["Turn left onto street %d." % i for i in range(10)] + ["Turn " + "x" * 300, "Turn left onto street 0.", "  "]
    r = post(client, ANDROID, texts).json()
    assert len(r["clips"]) == tc.MAX_TEXTS
    assert all(len(t) <= tc.MAX_CHARS for _, t in client.calls)


def test_concurrent_asks_synthesise_once(client, monkeypatch):
    async def run():
        vt = tc.voice_tuple(tc.Voice(**ANDROID))
        return await asyncio.gather(*[tc._clip(vt, "Keep right.") for _ in range(5)])
    out = asyncio.run(run())
    assert all(v is not None for v, _ in out)
    assert sum(1 for _, t in client.calls if t == "Keep right.") == 1


def test_memory_is_bounded():
    m = tc._MemLru(1000)
    for i in range(50):
        m.put("k%d" % i, b"x" * 100)
    assert m.bytes <= 1000 and len(m.d) == 10
    assert m.get("k0") is None and m.get("k49") is not None


def test_per_client_limit_serves_hits_but_does_not_synthesise(client):
    for i in range(20):
        post(client, ANDROID, ["Turn left onto %s%d." % (x, i) for x in "abcdef"])
    n = len(client.calls)
    assert n == tc.PER_CLIENT_LINES_PER_MIN            # 120 lines allowed, the rest refused
    r = post(client, ANDROID, ["Turn left onto a0.", "Turn left onto zz."]).json()  # hit served, nothing new
    assert [c["text"] for c in r["clips"]] == ["Turn left onto a0."] and len(client.calls) == n
    assert post(client, ANDROID, ["Keep right."], ip="5.6.7.8").json()["synthesised"] == 1


def test_a_forged_forwarded_for_does_not_escape_the_limit(client):
    for i in range(30):   # a new forged LEFT hop every time; the rightmost (Cloud Run's) stays the same
        post(client, ANDROID, ["Turn right onto %s%d." % (x, i) for x in "abcdef"], ip="10.0.0.%d, 9.9.9.9" % i)
    assert len(client.calls) == tc.PER_CLIENT_LINES_PER_MIN


def test_only_guidance_shaped_text_is_synthesised(client):
    r = post(client, ANDROID, ["Buy cheap watches now", "Hello, read my essay", "Make a sharp left."]).json()
    assert [c["text"] for c in r["clips"]] == ["Make a sharp left."]
    assert [t for _, t in client.calls] == ["Make a sharp left."]


def test_a_cancelled_first_asker_does_not_cancel_the_others(client):
    async def run():
        vt = tc.voice_tuple(tc.Voice(**ANDROID))
        first = asyncio.ensure_future(tc._clip(vt, "Turn left onto Shield Street."))
        await asyncio.sleep(0)
        others = [asyncio.ensure_future(tc._clip(vt, "Turn left onto Shield Street.")) for _ in range(3)]
        await asyncio.sleep(0)
        first.cancel()
        return await asyncio.gather(*others)
    out = asyncio.run(run())
    assert all(v is not None for v, _ in out)


def test_daily_synthesis_cap(client, monkeypatch):
    monkeypatch.setenv("TTS_CLIP_DAILY_SYNTH_CAP", "3")
    r = post(client, ANDROID, ["Turn left onto street %d." % i for i in range(5)]).json()
    assert r["synthesised"] == 3


def test_stats(client):
    post(client, ANDROID, ["Turn left."])
    post(client, ANDROID, ["Turn left."])
    s = client.get("/api/tts/clips/stats").json()
    assert s["hit_memory"] >= 1 and 0 < s["hit_rate"] <= 1
