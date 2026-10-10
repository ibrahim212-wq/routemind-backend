# -*- coding: utf-8 -*-
"""
Sekka's server-side skills (api/copilot_skills.py): departure planning on the prediction model, prayer times,
and "how long to X from here". Pure parts verified against known values; the I/O parts with their
dependencies stubbed.

Run:  python -m pytest tests/test_copilot_skills.py -q
"""

import asyncio
import json
import sys
import types
from datetime import date, datetime, timezone

import pytest

from api import copilot as base
from api import copilot_skills as sk
from api import copilot_v2 as v2

CAIRO = (30.0444, 31.2357)


def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


# ── prayer times: the Egyptian General Authority of Survey calendar, ±2 min ──────
@pytest.mark.parametrize("day,off,want", [
    # Cairo winter (UTC+2): a printed Egyptian calendar for 1 January
    (date(2026, 1, 1), 120, {"fajr": "5:18", "sunrise": "6:50", "dhuhr": "11:59", "asr": "14:47",
                             "maghrib": "17:08", "isha": "18:28"}),
    # Cairo midsummer under Egypt's summer time (UTC+3)
    (date(2026, 6, 21), 180, {"fajr": "4:08", "sunrise": "5:54", "dhuhr": "12:57", "asr": "16:33",
                              "maghrib": "19:59", "isha": "21:33"}),
])
def test_prayer_times_match_the_egyptian_calendar(day, off, want):
    got = sk.prayer_times(*CAIRO, day, off)
    for p, hhmm in want.items():
        assert abs(_mins(sk.clock(got[p])) - _mins(hhmm)) <= 2, (p, sk.clock(got[p]), hhmm)


def test_next_prayer_and_whether_the_drive_makes_it():
    ctx = {"user_lat": CAIRO[0], "user_lng": CAIRO[1], "utc_offset_min": 180, "eta_min": 25}
    # 2026-10-10 17:40 local = 14:40 UTC; Maghrib that day is ≈ 18:30
    a = sk.prayers_answer(ctx, now_utc=datetime(2026, 10, 10, 14, 40, tzinfo=timezone.utc))
    assert a["available"] and a["next"]["prayer"] == "maghrib" and a["next"]["arabic"] == "المغرب"
    assert 45 <= a["next"]["in_min"] <= 55 and a["arrives_before_next_prayer"] is True
    # after Isha the next one is tomorrow's Fajr
    b = sk.prayers_answer(ctx, now_utc=datetime(2026, 10, 10, 20, 30, tzinfo=timezone.utc))
    assert b["next"]["prayer"] == "fajr" and b["next"]["day"] == "tomorrow"
    # Friday noon is Jumu'ah
    c = sk.prayers_answer(ctx, now_utc=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc))
    assert c["next"]["prayer"] == "jumuah" and c["next"]["arabic"] == "صلاة الجمعة"
    # a specific one asked about
    d = sk.prayers_answer(ctx, "asr", now_utc=datetime(2026, 10, 10, 14, 40, tzinfo=timezone.utc))
    assert d["asked"]["prayer"] == "asr" and d["asked"]["passed"] is True
    assert sk.prayers_answer({})["available"] is False


# ── departure planning ─────────────────────────────────────────────────────────
def test_slots_and_the_best_window():
    now = datetime(2026, 10, 10, 16, 7)
    starts = sk.slot_starts(now, 2, "today")
    assert starts[0] == datetime(2026, 10, 10, 16, 15) and starts[-1] == datetime(2026, 10, 10, 18, 15)
    tom = sk.slot_starts(now, 2, "tomorrow")
    assert tom[0] == datetime(2026, 10, 11, 6, 0) and tom[-1] == datetime(2026, 10, 11, 23, 0)
    mins = [55, 52, 48, 41, 40, 41, 47, 50, 53]
    slots = list(zip(starts, mins))
    s = sk.summarize_plan(slots)
    assert s["best"]["leave"] == "17:15" and s["best"]["minutes"] == 40
    assert s["best_window"] == {"from": "17:00", "to": "17:30"}
    assert s["saving_vs_first_min"] == 15 and s["worst"]["minutes"] == 55


def _stub_model(monkeypatch, coverage, curve):
    tier1 = types.ModuleType("model.tier1")
    tier1.predict_route_duration = lambda jids, t, base_s: {"predicted_seconds": base_s * curve(t)}
    jm = types.ModuleType("services.junction_mapper")
    jm.COVERAGE_MIN_PCT = 0.70
    jm.calculate_route_coverage = lambda w: coverage
    jm.map_route_to_junctions = lambda w: [{"junction_id": "J1"}, {"junction_id": "J2"}]
    monkeypatch.setitem(sys.modules, "model.tier1", tier1)
    monkeypatch.setitem(sys.modules, "services.junction_mapper", jm)


ROUTE_CTX = {"route": [[31.23, 30.04], [31.30, 30.06], [31.40, 30.07]], "utc_offset_min": 180, "eta_min": 30}


def test_plan_departure_on_the_model(monkeypatch):
    # rush hour until 18:00, then it eases
    _stub_model(monkeypatch, 0.9, lambda t: 1.8 if t.hour < 18 else 1.1)
    out = asyncio.run(sk.plan_departure(ROUTE_CTX, "return_trip", 3, "today",
                                        now_utc=datetime(2026, 10, 10, 13, 50, tzinfo=timezone.utc)))
    assert out["available"] and out["trip"] == "return_trip"
    assert out["best"]["leave"] == "18:00" and out["saving_vs_first_min"] > 5


def test_plan_departure_says_so_outside_the_model(monkeypatch):
    _stub_model(monkeypatch, 0.3, lambda t: 1.0)
    out = asyncio.run(sk.plan_departure(ROUTE_CTX))
    assert out["available"] is False and "Cairo and Giza" in out["note"]
    assert asyncio.run(sk.plan_departure({}))["available"] is False


def test_eta_to_does_not_touch_the_trip(monkeypatch):
    async def resolve(ctx, q):
        return {"name": "Some Google Name", "lat": 29.96, "lng": 31.25, "distance_m": 12000}

    async def dur(coords, client):
        assert coords.startswith("31.2357,30.0444;31.25,29.96")
        return 1500
    monkeypatch.setattr(base, "_resolve_place", resolve)
    import api.places as places
    monkeypatch.setattr(places, "_mapbox_duration", dur)
    out = asyncio.run(sk.eta_to({"user_lat": 30.0444, "user_lng": 31.2357, "utc_offset_min": 180}, "المعادي"))
    assert out["found"] and out["minutes_from_here"] == 25 and out["requested"] == "المعادي"


# ── the engine offers and runs them ────────────────────────────────────────────
def test_the_model_can_call_the_skills_on_every_surface(monkeypatch):
    calls = []

    async def fake(messages, with_tools, tools=None):
        calls.append(tools)
        if len(calls) == 1:
            yield ("tool_calls", [{"id": "c1", "name": "prayer_times", "args": json.dumps({"which": "maghrib"})}])
        else:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs and json.loads(tool_msgs[-1]["content"])["asked"]["prayer"] == "maghrib"
            yield ("delta", "المغرب الساعة ستة ونص.")
            yield ("end", None)
    monkeypatch.setattr(base, "_stream_chat", fake)
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")

    class Req:
        messages = [{"role": "user", "content": "المغرب امتى؟"}]
        context = {"surface": "car", "can": ["show_overview"], "user_lat": 30.04, "user_lng": 31.23,
                   "utc_offset_min": 180}
        app_lang = "ar"
        prev_lang = "ar"
        pending_action = None
        limit_key = None

    async def go():
        return [json.loads(l) async for l in v2.stream_v2(Req())]
    lines = asyncio.run(go())
    names = {t["function"]["name"] for t in calls[0]}
    assert {"prayer_times", "plan_departure", "eta_to"} <= names
    assert any(l["t"] == "delta" for l in lines) and lines[-1]["t"] == "done"
