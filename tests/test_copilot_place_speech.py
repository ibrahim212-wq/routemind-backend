# -*- coding: utf-8 -*-
"""The copilot never SPEAKS Google place content (api/copilot_place_speech.py; cost Phase 4, flagged).

With the flag off (the default) the model never receives a place's name, rating, reviews, price, hours, phone or
address — from the place tools or from the trip context — so its spoken reply cannot carry them; the client's
actions (cards / pins) still do. COPILOT_SPEAK_PLACE_CONTENT=1 restores the previous behaviour.

Run:  python -m pytest tests/test_copilot_place_speech.py -q
"""
from api import copilot_place_speech as ps

FIND = {"found": True, "where": "along_route", "count": 2,
        "places": [{"name": "Café Riche", "id": "ChIJa", "rating": 4.6, "reviews": 812, "price": 2,
                    "open_now": True, "away": "2 km"},
                   {"name": "Zooba", "id": "ChIJb", "detour_min": 3, "open_now": False}],
        "note": "Shown on the map."}
DETAILS = {"found": True, "place": "Café Riche", "rating": 4.6, "reviews": 812, "hours": ["Mon: 9–1"],
           "editorial": "Historic café.", "review_samples": [{"text": "Lovely"}], "phone": "+20 2 2392 9793",
           "note": "Summarize helpfully."}


def test_find_places_keeps_only_order_ids_and_our_distances(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)
    out = ps.for_model("find_places", FIND)
    assert out["places"] == [{"n": 1, "id": "ChIJa", "away": "2 km"}, {"n": 2, "id": "ChIJb", "detour_min": 3}]
    assert out["count"] == 2 and out["found"] is True
    blob = repr(out)
    for s in ("Café Riche", "Zooba", "4.6", "812", "open_now", "price"):
        assert s not in blob
    assert "never spoken" in out["note"]


def test_place_details_carries_no_content(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)
    out = ps.for_model("place_details", DETAILS)
    assert set(out) == {"found", "note"}


def test_the_flag_restores_everything(monkeypatch):
    monkeypatch.setenv("COPILOT_SPEAK_PLACE_CONTENT", "1")
    assert ps.for_model("find_places", FIND) is FIND
    assert ps.context_for_model({"dest_name": "Cloud 9"})["dest_name"] == "Cloud 9"


def test_tools_without_place_content_are_untouched(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)
    r = {"found": True, "name": "switch", "commit": "done"}
    assert ps.for_model("switch_route", r) is r


def test_context_names_are_withheld(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)
    c = ps.context_for_model({"dest_name": "Cloud 9 Mall", "stops": [{"name": "Zooba"}, "ATM"], "user_lat": 30.0})
    assert c["dest_name"] == "set (shown on the screen)"
    assert c["stops"] == ["stop 1", "stop 2"]
    assert c["user_lat"] == 30.0
