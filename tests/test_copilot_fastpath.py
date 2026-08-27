# -*- coding: utf-8 -*-
"""
Fast-path accuracy suite — Phase-7 "trip-context accuracy" requirement:
the assistant's spoken numbers must EQUAL the source-of-truth numbers exactly.
The fast-path is the deterministic layer that guarantees it for the most
common trip questions; these tests assert both the matching precision and the
value fidelity against a reference context.
"""

import re

import pytest

from api.copilot_fastpath import match_intent, answer, try_fastpath

CTX = {
    "dest_name": "Mall of Arabia",
    "remaining_km": 23.4,
    "eta_min": 31,
    "speed_kmh": 87,
    "speed_limit_kmh": 80,
    "current_road": "Ring Road",
    "next_maneuver": "Exit toward 26th of July Corridor",
    "next_maneuver_m": 900,
    "local_time": "17:40",
    "cameras_ahead": [
        {"type": "speed", "distance_ahead_km": 2.1, "limit_kmh": 80},
        {"type": "speed", "distance_ahead_km": 9.0, "limit_kmh": 90},
        {"type": "red_light", "distance_ahead_km": 14.2},
    ],
}


# ── Matching: the exact driver phrasings route to the right intent ───────────
@pytest.mark.parametrize("text,intent", [
    ("كام رادار قدامي؟", "cameras_count"),
    ("how many cameras ahead", "cameras_count"),
    ("فين الرادار الجاي", "next_camera"),
    ("where is the next camera", "next_camera"),
    ("what's the speed limit", "speed_limit"),
    ("السرعة القصوى كام", "speed_limit"),
    ("سرعتي كام", "current_speed"),
    ("how fast am i going", "current_speed"),
    ("انا في شارع ايه", "current_road"),
    ("what road am i on", "current_road"),
    ("what's my next turn", "next_turn"),
    ("هوصل امتى", "arrival_time"),
    ("what time will i arrive", "arrival_time"),
    ("فاضل كام", "eta_remaining"),
    ("how long left", "eta_remaining"),
    ("how far to go", "distance_remaining"),
    ("فاضل كام كيلو", "distance_remaining"),
])
def test_intent_matches(text, intent):
    assert match_intent(text) == intent


# ── Precision guards: these must NOT fast-path ───────────────────────────────
@pytest.mark.parametrize("text", [
    "how long is the traffic jam",            # traffic → model
    "الزحمة هتاخد كام",                        # traffic → model
    "add the nearest gas station",            # action
    "ضيفلي أقرب بنزينة",                       # action
    "how many cameras ahead and where is the nearest pharmacy",  # multi/actiony
    "كام رادار وفين اقرب صيدلية",              # multi/actiony
    "take me home",                           # action
    "is this normal for this time",           # baseline → traffic_check tool
    "tell me about the history of the ring road while we drive there today",
])
def test_no_fastpath_for(text):
    assert match_intent(text) is None


# ── Value fidelity: spoken == source, exactly ────────────────────────────────
def test_cameras_count_exact():
    en = answer("cameras_count", CTX, "en")
    ar = answer("cameras_count", CTX, "ar")
    assert en.startswith("3 cameras ahead")
    assert "limit 80" in en
    assert "3 رادارات" in ar and "80" in ar


def test_cameras_zero_honest():
    ctx = dict(CTX, cameras_ahead=[])
    assert "No speed cameras" in answer("cameras_count", ctx, "en")
    assert "مفيش رادارات" in answer("cameras_count", ctx, "ar")


def test_cameras_unknown_falls_to_model():
    ctx = {k: v for k, v in CTX.items() if k != "cameras_ahead"}
    assert answer("cameras_count", ctx, "en") is None


def test_next_camera_distance_from_source():
    en = answer("next_camera", CTX, "en")
    # 2.1 km → spoken as "2 kilometers" (rounded to halves, no decimals)
    assert "2 kilometers" in en and "limit 80" in en
    ar = answer("next_camera", CTX, "ar")
    assert "80" in ar


def test_speed_limit_and_over_limit_flag():
    en = answer("speed_limit", CTX, "en")
    assert "80" in en and "over" in en          # 87 > 80+3 → warn
    slower = dict(CTX, speed_kmh=78)
    assert "over" not in answer("speed_limit", slower, "en")


def test_speed_limit_missing_is_honest():
    ctx = {k: v for k, v in CTX.items() if k != "speed_limit_kmh"}
    assert "No posted speed-limit data" in answer("speed_limit", ctx, "en")
    assert "مفيش" in answer("speed_limit", ctx, "ar")


def test_eta_exact_clock_math():
    en = answer("eta_remaining", CTX, "en")
    # 17:40 + 31 min = 18:11
    assert "31 minutes" in en and "18:11" in en
    ar = answer("arrival_time", CTX, "ar")
    assert "18:11" in ar


def test_distance_remaining_rounded_for_ear():
    en = answer("distance_remaining", CTX, "en")
    assert "23" in en and not re.search(r"\d+\.\d", en)


def test_current_road_verbatim():
    assert "Ring Road" in answer("current_road", CTX, "en")
    assert "Ring Road" in answer("current_road", CTX, "ar")


def test_next_turn_carries_instruction_and_distance():
    en = answer("next_turn", CTX, "en")
    assert "Exit toward 26th of July Corridor" in en
    # 900 m → "900 meters"
    assert "900 meters" in en


def test_entrypoint_guards():
    assert try_fastpath("كام رادار قدامي", CTX, "ar", False) is not None
    assert try_fastpath("كام رادار قدامي", CTX, "ar", True) is None   # pending action
    assert try_fastpath("كام رادار قدامي", {}, "ar", False) is None   # no context
