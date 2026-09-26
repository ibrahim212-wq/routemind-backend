# -*- coding: utf-8 -*-
"""The copilot never SPEAKS Google place content (api/copilot_place_speech.py; cost Phase 4, flagged).

With the flag off (the default): the model never receives a place's rating, reviews, summary, price, hours,
phone or address; it keeps names and ids (so "add the second one" still names the right place to the tool); and
the one output gate scrubs every Google place name the turn has seen — from tool results and actions, the trip
context, the pending action and the "[Shown: …]" history — out of the spoken text. COPILOT_SPEAK_PLACE_CONTENT=1
restores the previous behaviour.

Run:  python -m pytest tests/test_copilot_place_speech.py -q
"""
import asyncio
import json

from api import copilot_place_speech as ps
from api import copilot_v2 as v2
from tests.test_copilot_v2_stream import FakeReq, deltas, run_turn

FIND = {"found": True, "where": "along_route", "count": 2,
        "places": [{"name": "Café Riche", "id": "ChIJa", "rating": 4.6, "reviews": 812, "price": 2,
                    "open_now": True, "away": "2 km"},
                   {"name": "Zooba", "id": "ChIJb", "detour_min": 3, "open_now": False}],
        "note": "Shown on the map."}
DETAILS = {"found": True, "place": "Café Riche", "rating": 4.6, "reviews": 812, "hours": ["Mon: 9–1"],
           "editorial": "Historic café.", "review_samples": [{"text": "Lovely"}], "phone": "+20 2 2392 9793",
           "note": "Summarize helpfully."}


def test_the_model_keeps_names_and_ids_never_the_descriptive_content(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)
    out = ps.for_model("find_places", FIND)
    assert out["places"] == [{"name": "Café Riche", "id": "ChIJa", "away": "2 km"},
                             {"name": "Zooba", "id": "ChIJb", "detour_min": 3}]
    d = ps.for_model("place_details", DETAILS)
    assert set(d) == {"found", "place", "note"}
    assert "never spoken" in d["note"]
    assert "number" not in ps.for_model("pending_action", {"type": "dial", "number": "010", "place": "X"})


def test_names_are_collected_from_results_actions_context_and_history():
    names = ps.names_in(FIND) | ps.names_in({"type": "add_stop", "place": {"name": "Zooba 2", "lat": 1}})
    assert {"Café Riche", "Zooba", "Zooba 2"} <= names
    ctx = {"dest_name": "Cloud 9 Mall", "stops": [{"name": "Wadi Degla"}],
           "alternatives": [{"leg_to": "City Stars"}], "user_lat": 30.0}
    assert ps.context_names(ctx) == {"Cloud 9 Mall", "Wadi Degla", "City Stars"}
    assert ps.names_in({"dest_name": "Home"}) == set()          # our own labels are never scrubbed
    hist = [{"role": "user", "content": "[Shown: 1. Café Riche, 2. Zooba]"}]
    assert ps.names_in_history(hist) == {"Café Riche", "Zooba"}


def test_scrub_replaces_names_in_both_languages():
    assert ps.scrub("Café Riche is 2 km ahead.", {"Café Riche"}, "en") == "that place is 2 km ahead."
    assert ps.scrub("كافيه ريش على بعد ٢ كيلو", {"كافيه ريش"}, "ar") == "المكان ده على بعد ٢ كيلو"


def test_the_flag_restores_everything(monkeypatch):
    monkeypatch.setenv("COPILOT_SPEAK_PLACE_CONTENT", "1")
    assert ps.for_model("find_places", FIND) is FIND
    assert ps.rule() == []


def test_a_name_the_model_says_anyway_is_never_spoken(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)

    async def fake_tool(name, args, ctx):
        return FIND, {"type": "show_places", "places": FIND["places"], "display": "cards",
                      "requires_confirm": False}
    monkeypatch.setattr(v2, "execute_tool_v2", fake_tool)
    script = [
        [("tool_calls", [{"id": "c1", "name": "find_places", "args": json.dumps({"query": "cafe"})}])],
        [("delta", "I found two. Café Riche is the first one, 2 km ahead. Zooba is the second.")],
    ]
    lines, calls = asyncio.run(run_turn(FakeReq("find me a cafe on the way",
                                                ctx={"dest_name": "Cloud 9 Mall"}), script, monkeypatch))
    spoken = deltas(lines)
    assert "Café Riche" not in spoken and "Zooba" not in spoken
    assert "2 km ahead" in spoken
    # the model saw the names (for the next tool) but no rating, and the rule
    sent = json.dumps(calls[-1]["messages"], ensure_ascii=False)
    assert "Café Riche" in sent and '"rating"' not in sent
    assert "SPEAKING PLACES" in sent
    # the cards still carry everything
    assert [l for l in lines if l["t"] == "action"][0]["action"]["places"][0]["rating"] == 4.6


def test_the_destination_name_is_scrubbed(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)
    script = [[("delta", "You'll reach Cloud 9 Mall in 12 minutes.")]]
    lines, _ = asyncio.run(run_turn(FakeReq("how long left", ctx={"dest_name": "Cloud 9 Mall"}), script,
                                    monkeypatch))
    assert "Cloud 9" not in deltas(lines)
    assert "12 minutes" in deltas(lines)
