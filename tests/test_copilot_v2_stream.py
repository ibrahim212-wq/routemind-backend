# -*- coding: utf-8 -*-
"""
Stream-level tests for the CopilotV2 turn engine (api/copilot_v2.py) with a
SCRIPTED model — proves the end-to-end guarantees:

  • the resolved language is emitted first ({"t":"meta"}) and every delta the
    client receives matches it, whatever the model produced
  • wrong-language generations are regenerated / translated, never emitted
  • the fact fast-path answers without any model call
  • multi-tool rounds execute in order with at most one confirm-class action
  • scenario paths: tool failure, no-results, empty generation

Run:  python -m pytest tests/test_copilot_v2_stream.py -q
"""

import asyncio
import json
import types

import pytest

from api import copilot as base
from api import copilot_v2 as v2
from api.copilot_lang import reply_lang_ok


class FakeReq:
    def __init__(self, text, ctx=None, prev_lang=None, app_lang="en",
                 pending_action=None, history=None):
        msgs = list(history or [])
        msgs.append({"role": "user", "content": text})
        self.messages = msgs
        self.context = ctx or {}
        self.app_lang = app_lang
        self.pending_action = pending_action
        self.prev_lang = prev_lang
        self.v2 = True


def scripted_stream_chat(script):
    """Replace base._stream_chat with a generator replaying `script` — a list
    of per-call event lists. Each event: ("delta", text) | ("tool_calls", [...])
    | ("end", None). Records every call's (messages, with_tools, tools)."""
    calls = []

    async def fake(messages, with_tools, tools=None):
        idx = len(calls)
        calls.append({"messages": [dict(m) for m in messages],
                      "with_tools": with_tools, "tools": tools})
        events = script[min(idx, len(script) - 1)]
        emitted_tools = False
        for kind, payload in events:
            if kind == "tool_calls":
                emitted_tools = True
            yield (kind, payload)
        if not emitted_tools and (not events or events[-1][0] != "end"):
            yield ("end", None)

    return fake, calls


async def run_turn(req, script, monkeypatch, openai_key="test"):
    fake, calls = scripted_stream_chat(script)
    monkeypatch.setattr(base, "_stream_chat", fake)
    monkeypatch.setattr(base, "OPENAI_KEY", openai_key)
    lines = []
    async for raw in v2.stream_v2(req):
        lines.append(json.loads(raw))
    return lines, calls


def deltas(lines):
    return "".join(l["text"] for l in lines if l["t"] == "delta")


def actions(lines):
    return [l["action"] for l in lines if l["t"] == "action"]


def done(lines):
    for l in lines:
        if l["t"] == "done":
            return l
    return None


# ═══════════════════════════════════════════════════════════════════════════

def test_meta_line_first_and_lang_pinned(monkeypatch):
    req = FakeReq("عايز أعرف الدنيا عاملة ايه")
    script = [[("delta", "الدنيا تمام يا صاحبي، الطريق قدامك فاضي. ")]]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    assert lines[0]["t"] == "meta" and lines[0]["lang"] == "ar"
    assert done(lines)["lang"] == "ar"
    assert reply_lang_ok(deltas(lines), "ar")


def test_happy_path_english(monkeypatch):
    req = FakeReq("tell me something nice")
    script = [[("delta", "The road ahead is clear, "),
               ("delta", "smooth sailing all the way.")]]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    assert lines[0]["lang"] == "en"
    assert "smooth sailing" in deltas(lines)


def test_wrong_language_regenerated_never_emitted(monkeypatch):
    """Model answers English to an Arabic turn → first draft discarded, the
    corrective regeneration's Arabic is what the client sees."""
    req = FakeReq("احكيلي عن الطريق ده")
    script = [
        [("delta", "This road was built in 1985 and is very famous. ")],   # bad
        [("delta", "الطريق ده اتبنى من زمان وهو من أشهر طرق القاهرة. ")],  # regen
    ]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    text = deltas(lines)
    assert "1985 and is very famous" not in text
    assert reply_lang_ok(text, "ar")
    assert len(calls) == 2
    # the regen carried the hard corrective instruction
    joined = " ".join(m.get("content") or "" for m in calls[1]["messages"]
                      if m["role"] == "system")
    assert "WRONG LANGUAGE" in joined


def test_double_failure_falls_to_translation(monkeypatch):
    """Both drafts wrong → deterministic translation pass output is emitted."""
    req = FakeReq("قولي حاجة عن مصر")
    script = [
        [("delta", "Egypt is a beautiful country with great history. ")],  # bad
        [("delta", "Egypt is truly wonderful, you know. ")],               # regen bad
        [("delta", "مصر جميلة بجد وتاريخها عظيم. ")],                      # translation
    ]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    text = deltas(lines)
    assert reply_lang_ok(text, "ar")
    assert "beautiful country" not in text
    assert "مصر" in text
    assert len(calls) == 3


def test_midstream_wrong_sentence_translated_inline(monkeypatch):
    req = FakeReq("كمل كلام معايا")
    script = [
        [("delta", "ماشي يا صاحبي، الطريق حلو قدامنا. "),
         ("delta", "By the way traffic is completely clear ahead of us now. ")],
        [("delta", "وبالمناسبة الطريق فاضي قدامنا خالص. ")],   # inline translation
    ]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    text = deltas(lines)
    assert "completely clear" not in text
    assert reply_lang_ok(text, "ar")


def test_fastpath_answers_without_model(monkeypatch):
    ctx = {"cameras_ahead": [{"type": "speed", "distance_ahead_km": 2.0,
                              "limit_kmh": 80}],
           "local_time": "12:00"}
    req = FakeReq("كام رادار قدامي؟", ctx=ctx)
    script = [[("delta", "SHOULD NEVER BE CALLED")]]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    assert len(calls) == 0, "fast-path must not call the model"
    text = deltas(lines)
    assert "رادار" in text and "80" in text
    assert done(lines)["expects_reply"] is False


def test_tool_round_action_emitted_and_followup_spoken(monkeypatch):
    """avoid_jam with a faster alternative → switch_route action (commit=done)
    + follow-up speech in the pinned language."""
    ctx = {"alternatives": [
        {"index": 1, "delta_min": -6, "via": "Ring Road"},
        {"index": 2, "delta_min": 3, "via": "Autostrad"}],
        "traffic_segments": [{"level": "heavy", "distance_ahead_km": 2.0}]}
    req = FakeReq("عديني من الزحمة دي", ctx=ctx)
    script = [
        [("tool_calls", [{"id": "c1", "name": "avoid_jam", "args": "{}"}])],
        [("delta", "اتحولت على الدائري — هتكسب حوالي 6 دقايق. ")],
    ]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    acts = actions(lines)
    assert len(acts) == 1
    assert acts[0]["type"] == "switch_route"
    assert acts[0]["index"] == 1
    assert acts[0]["commit"] == "done"
    assert "6" in deltas(lines)
    # second pass got the tool result + the language pin again
    sys_msgs = [m.get("content") or "" for m in calls[1]["messages"]
                if m["role"] == "system"]
    assert any("Tool result" in s for s in sys_msgs)


def test_avoid_jam_honest_when_no_better_route(monkeypatch):
    ctx = {"alternatives": [{"index": 1, "delta_min": 4, "via": "Autostrad"}],
           "traffic_segments": [{"level": "heavy", "distance_ahead_km": 2.0}],
           "traffic_delay_min": 7}
    req = FakeReq("avoid this jam please", ctx=ctx)
    script = [
        [("tool_calls", [{"id": "c1", "name": "avoid_jam", "args": "{}"}])],
        [("delta", "You're already on the best route — the jam costs about "
                   "7 minutes either way. ")],
    ]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    assert actions(lines) == []          # no switch happened
    tool_msg = [m for m in calls[1]["messages"] if m["role"] == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert payload["no_better_route"] is True
    assert payload["current_delay_min"] == 7


def test_multi_tool_round_confirm_cap(monkeypatch):
    """Two confirm-class tools in one round: the first arms, the second is
    blocked with an explanatory result — never two pending confirmations."""
    ctx = {"user_lat": 30.0, "user_lng": 31.2,
           "saved_places": {"home": {"name": "Home", "lat": 30.1, "lng": 31.3},
                            "work": {"name": "Work", "lat": 30.2, "lng": 31.4}}}
    req = FakeReq("وديني البيت ولا الشغل مش عارف", ctx=ctx)
    script = [
        [("tool_calls", [
            {"id": "c1", "name": "navigate_saved", "args": '{"place":"home"}'},
            {"id": "c2", "name": "navigate_saved", "args": '{"place":"work"}'}])],
        [("delta", "جهزتلك مسار البيت — أأكد؟ ")],
    ]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    acts = actions(lines)
    assert len(acts) == 1 and acts[0]["type"] == "change_destination"
    assert acts[0]["place"]["name"] == "Home"
    tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    second = json.loads(tool_msgs[1]["content"])
    assert second.get("blocked") is True
    assert done(lines)["expects_reply"] is True


def test_skip_next_stop_and_reminder(monkeypatch):
    ctx = {"stops": [{"name": "Master"}, {"name": "El Ezaby"}], "eta_min": 40}
    req = FakeReq("skip the next stop and remind me 10 minutes before arrival",
                  ctx=ctx)
    script = [
        [("tool_calls", [
            {"id": "c1", "name": "skip_next_stop", "args": "{}"},
            {"id": "c2", "name": "remind_before_arrival",
             "args": '{"minutes_before":10}'}])],
        [("delta", "Skipped Master, and I'll remind you ten minutes out. ")],
    ]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    acts = actions(lines)
    assert acts[0]["type"] == "remove_stop" and acts[0]["next"] is True
    assert acts[0]["name"] == "Master"
    assert acts[1]["type"] == "remind_before_arrival"
    assert acts[1]["minutes_before"] == 10


def test_switch_route_fastest_selector(monkeypatch):
    ctx = {"alternatives": [{"index": 1, "delta_min": 2, "via": "A"},
                            {"index": 2, "delta_min": -5, "via": "Ring Road"}]}
    req = FakeReq("خدني أسرع طريق", ctx=ctx)
    script = [
        [("tool_calls", [{"id": "c1", "name": "switch_route",
                          "args": '{"selector":"fastest"}'}])],
        [("delta", "حولتك على الدائري — أسرع بحوالي 5 دقايق. ")],
    ]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    acts = actions(lines)
    assert acts[0]["type"] == "switch_route" and acts[0]["index"] == 2


def test_switch_route_fastest_honest_when_already_best(monkeypatch):
    ctx = {"alternatives": [{"index": 1, "delta_min": 4, "via": "A"}]}
    req = FakeReq("take the fastest route", ctx=ctx)
    script = [
        [("tool_calls", [{"id": "c1", "name": "switch_route",
                          "args": '{"selector":"fastest"}'}])],
        [("delta", "You're already on the fastest route. ")],
    ]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    assert actions(lines) == []
    payload = json.loads([m for m in calls[1]["messages"]
                          if m["role"] == "tool"][0]["content"])
    assert payload.get("already_fastest") is True


def test_say_it_once_preamble_then_identical_followup(monkeypatch):
    """PRODUCTION DEFECT 2026-08-27: a pass can emit a full answer AND a tool
    call; the post-tool pass then repeats it verbatim and the driver hears the
    whole reply twice. The dedup guard must emit it exactly once."""
    answer = "قدامك زحمة تقيلة بعد حوالي ٢ كيلو على صلاح سالم، وهتأخرك ٦ دقايق. "
    req = FakeReq("الطريق قدامي عامل ايه؟", ctx={"user_lat": 30.0,
                                                 "user_lng": 31.2,
                                                 "route": [[31.2, 30.0],
                                                           [31.3, 30.1]]})
    script = [
        # pass 1: speaks the whole answer AND calls an action-less tool
        [("delta", answer),
         ("tool_calls", [{"id": "c1", "name": "traffic_check", "args": "{}"}])],
        # pass 2: the follow-up says exactly the same thing
        [("delta", answer)],
    ]

    async def fake_check(ctx):
        return {"available": False, "note": "no coverage"}
    monkeypatch.setattr(v2, "_traffic_check", fake_check)

    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    text = deltas(lines)
    assert text.count("صلاح سالم") == 1, f"answer spoken twice: {text!r}"
    assert "زحمة تقيلة" in text


def test_dedup_allows_genuine_short_acks(monkeypatch):
    """Short lines are exempt — two 'تمام.'s in one turn are legitimate."""
    req = FakeReq("تمام")
    script = [[("delta", "تمام. "), ("delta", "تمام. ")]]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    assert deltas(lines).count("تمام") == 2


def test_dedup_does_not_suppress_distinct_sentences(monkeypatch):
    req = FakeReq("احكيلي عن الطريق")
    script = [[("delta", "الطريق قدامك فاضي تمامًا دلوقتي. "),
               ("delta", "بس فيه رادار بعد كيلومترين خد بالك منه. ")]]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    text = deltas(lines)
    assert "فاضي" in text and "رادار" in text


def test_empty_generation_falls_back_localized(monkeypatch):
    req = FakeReq("تمام")
    script = [[("end", None)]]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    assert deltas(lines).strip() == "تمام."


def test_unconfigured_backend_errors_cleanly(monkeypatch):
    req = FakeReq("hello")
    lines, _ = asyncio.run(run_turn(req, [[("end", None)]], monkeypatch,
                                    openai_key=""))
    assert lines[0]["t"] == "error"


def test_where_parked_zooms_to_spot(monkeypatch):
    ctx = {"parking": {"lat": 30.05, "lng": 31.2, "age_min": 95,
                       "distance_km": 12.0}}
    req = FakeReq("انا راكن فين؟", ctx=ctx)
    script = [
        [("tool_calls", [{"id": "c1", "name": "where_parked", "args": "{}"}])],
        [("delta", "عربيتك متركونة من ساعة ونص تقريبًا — مبينهالك على الخريطة. ")],
    ]
    lines, _ = asyncio.run(run_turn(req, script, monkeypatch))
    acts = actions(lines)
    assert acts[0]["type"] == "zoom_to_place"
    assert acts[0]["point"]["lat"] == 30.05


def test_where_parked_honest_when_none(monkeypatch):
    req = FakeReq("where did I park?", ctx={"user_lat": 30.0, "user_lng": 31.0})
    script = [
        [("tool_calls", [{"id": "c1", "name": "where_parked", "args": "{}"}])],
        [("delta", "There's no parking spot saved yet — say save my parking "
                   "when you park. ")],
    ]
    lines, calls = asyncio.run(run_turn(req, script, monkeypatch))
    assert actions(lines) == []
    payload = json.loads([m for m in calls[1]["messages"]
                          if m["role"] == "tool"][0]["content"])
    assert payload["found"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Context formatter honesty + sun math
# ═══════════════════════════════════════════════════════════════════════════

def test_context_empty_is_honest():
    s = v2._format_context_v2({})
    assert "No live trip data" in s


def test_context_over_limit_and_schedule():
    s = v2._format_context_v2({
        "speed_kmh": 95, "speed_limit_kmh": 80, "schedule_delta_min": 7,
        "cameras_ahead": [], "dest_name": "X", "remaining_km": 5})
    assert "OVER the limit" in s
    assert "BEHIND" in s
    assert "none on the remaining route" in s


def test_context_gps_stale_flag():
    s = v2._format_context_v2({"gps_age_s": 42, "dest_name": "X"})
    assert "GPS fix is 42 s old" in s


def test_context_camera_count_is_exact():
    cams = [{"type": "speed", "distance_ahead_km": k, "limit_kmh": 80}
            for k in (1.0, 3.5, 7.0, 9.9, 14.2, 18.0, 22.5)]
    s = v2._format_context_v2({"cameras_ahead": cams, "dest_name": "X"})
    assert "Speed cameras ahead: 7 total" in s


def test_context_schedule_ahead_and_bridges():
    s = v2._format_context_v2({
        "dest_name": "X", "schedule_delta_min": -6,
        "bridges_ahead": {"count": 2, "next_km": 1.5, "next_action": "take it"},
        "saved_places": {"home": {"name": "Home", "lat": 1, "lng": 2}},
        "parking": {"lat": 30.0, "lng": 31.0, "age_min": 40}})
    assert "AHEAD of the original ETA" in s
    assert "Bridges/flyovers ahead: 2" in s and "take it" in s
    assert "Saved places available: home" in s
    assert "Car parked at a saved spot (40 min ago)" in s


def test_sun_times_cairo_sane():
    from datetime import datetime, timezone
    # Aug 27, 12:00 UTC (15:00 Cairo) — daylight; sunset ~18:2x local
    r = v2._sun_times(30.05, 31.23, 180,
                      datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc))
    assert r is not None
    rise, set_, night = r
    rh = int(rise.split(":")[0]); sh = int(set_.split(":")[0])
    assert 4 <= rh <= 7
    assert 17 <= sh <= 19
    assert night is False
    # 20:00 Cairo → night
    r2 = v2._sun_times(30.05, 31.23, 180,
                       datetime(2026, 8, 27, 17, 0, tzinfo=timezone.utc))
    assert r2[2] is True
