# -*- coding: utf-8 -*-
"""Per-user daily limits (api/usage_limits.py; cost Phase 4 lever L).

  • the counter allows exactly `limit` per key per Cairo day and resets at Cairo midnight
  • the key is the app's device id; an old client is keyed by its address, never unlimited
  • a copilot turn that reaches the model is refused with an in-stream error line both apps already speak
    (code daily_limit, the spoken text in the turn's language); fast-path turns are not counted
  • Plan-a-Drive's stream answers 429 once the day's plans are used

Run:  python -m pytest tests/test_usage_limits.py -q
"""
import asyncio
import json
from datetime import datetime

import pytest

from api import copilot as base
from api import copilot_v2 as v2
from api import usage_limits as ul


class FakeReq:
    def __init__(self, text, limit_key="dev:test-device-1", app_lang="en"):
        self.messages = [{"role": "user", "content": text}]
        self.context = {}
        self.app_lang = app_lang
        self.pending_action = None
        self.prev_lang = None
        self.limit_key = limit_key


def _collect(req):
    async def run():
        return [json.loads(l) async for l in v2.stream_v2(req)]
    return asyncio.run(run())


def test_counter_allows_exactly_the_limit_and_resets_at_cairo_midnight():
    c = ul.DailyCounter()
    d1 = datetime(2026, 9, 26, 23, 50, tzinfo=ul._CAIRO)
    assert all(c.take("k", 3, d1) for _ in range(3))
    assert not c.take("k", 3, d1)
    assert c.take("other", 3, d1)  # per key
    d2 = datetime(2026, 9, 27, 0, 5, tzinfo=ul._CAIRO)
    assert c.take("k", 3, d2)


def test_user_key_prefers_the_device_id_and_falls_back_to_the_address():
    assert ul.user_key({"x-routemind-device": "3f2c1d7e-aaaa-bbbb-cccc-1234567890ab"}, "1.2.3.4") \
        == "dev:3f2c1d7e-aaaa-bbbb-cccc-1234567890ab"
    assert ul.user_key({}, "1.2.3.4", device_id="abcdefgh12") == "dev:abcdefgh12"
    assert ul.user_key({"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, "10.0.0.2") == "ip:10.0.0.1"
    assert ul.user_key({"x-routemind-device": "short"}, "10.0.0.2") == "ip:10.0.0.2"


def test_a_model_turn_beyond_the_limit_is_an_in_stream_error_in_the_turn_language(monkeypatch):
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    monkeypatch.setenv("COPILOT_TURNS_PER_DAY", "0")
    monkeypatch.setattr(v2, "COPILOT_TURNS", ul.DailyCounter())
    lines = _collect(FakeReq("find me a quiet cafe on the way"))
    assert lines[0]["t"] == "meta"
    err = [l for l in lines if l["t"] == "error"]
    assert err and err[0]["code"] == "daily_limit"
    assert err[0]["spoken"] == "That's today's limit for Sekka. I'm back tomorrow."
    lines_ar = _collect(FakeReq("عايز كافيه هادي في الطريق", app_lang="ar"))
    err_ar = [l for l in lines_ar if l["t"] == "error"][0]
    assert err_ar["lang"] == "ar" and "النهارده" in err_ar["spoken"]


def test_fast_path_turns_are_never_counted(monkeypatch):
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    monkeypatch.setenv("COPILOT_TURNS_PER_DAY", "0")
    counter = ul.DailyCounter()
    monkeypatch.setattr(v2, "COPILOT_TURNS", counter)
    lines = _collect(FakeReq("mute"))
    assert not [l for l in lines if l["t"] == "error"]
    assert counter.used("dev:test-device-1") == 0


def test_no_key_no_limit_for_direct_engine_callers(monkeypatch):
    # stream_v2 called without a key (tests / internal callers) is not limited
    monkeypatch.setattr(base, "OPENAI_KEY", "")
    lines = _collect(FakeReq("hello", limit_key=None))
    assert [l for l in lines if l["t"] == "error"][0]["code"] == "not_configured"


def test_plan_stream_answers_429_once_the_days_plans_are_used(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import plan_drive_stream as pds

    monkeypatch.setenv("PLANS_PER_DAY", "0")
    monkeypatch.setattr(ul, "PLANS", ul.DailyCounter())
    app = FastAPI()
    app.include_router(pds.router)
    body = {"waypoints": [], "target_date": "2026-09-27", "base_duration_seconds": 900,
            "origin_lat": 30.0, "origin_lng": 31.2, "dest_lat": 30.1, "dest_lng": 31.3}
    r = TestClient(app).post("/plan-drive-stream", json=body,
                             headers={"X-RouteMind-Device": "3f2c1d7e-aaaa-bbbb-cccc-1234567890ab"})
    assert r.status_code == 429
    assert r.json()["detail"]["error"] == "daily_limit"
