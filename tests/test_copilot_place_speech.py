# -*- coding: utf-8 -*-
"""The copilot does not SPEAK Google place content (api/copilot_place_speech.py; cost Phase 4, flagged).

With the flag off (the default): the model never receives a place's rating, reviews, summary, price, hours,
phone or address; it keeps names and ids (so "add the second one" still names the right place to the tool); the
one output gate scrubs the Google place names the turn has seen (place tools, pending action, "[Shown: …]" cards)
out of the spoken text — whole words, with their Egyptian form, leading segment and abbreviation parts — and never
touches road names, what the user asked for, our labels, the destination / stops, or generic words.
COPILOT_SPEAK_PLACE_CONTENT=1 restores the previous behaviour.

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
    assert "number" not in ps.for_model("pending_action", {"type": "dial", "number": "010", "place": "X"})


def test_names_come_only_from_google_sourced_fields():
    assert ps.google_names("find_places", FIND) == {"Café Riche", "Zooba"}
    assert ps.google_names("add_stop", {"found": False, "requested": "gas station",
                                        "other_candidates": ["Wataniya Maadi"]}) == {"Wataniya Maadi"}
    assert ps.google_names("reroute_via", {"via": "Zooba Zamalek"}) == {"Zooba Zamalek"}
    # a route summary is a Mapbox road name, not a place
    assert ps.google_names("avoid_jam", {"via": "Ring Road"}, {"via": "Ring Road"}) == set()
    assert ps.google_names("switch_route", {"via": "26th of July Corridor"}) == set()
    # our labels and single generic words never
    assert ps.google_names("change_destination", {"new_destination": "Home"}) == set()
    assert ps.google_names("find_places", {"places": [{"name": "Total"}, {"name": "Mall"}]}) == set()
    hist = [{"role": "user", "content": "[Shown: 1. Total, Nasr City, 2. Zooba]"}]
    assert ps.names_in_history(hist) == {"Total, Nasr City", "Zooba"}


def test_scrub_whole_words_variants_and_both_languages():
    assert ps.scrub("Café Riche is 2 km ahead.", {"Café Riche"}, "en") == "that place is 2 km ahead."
    assert ps.scrub("كافيه ريش على بعد ٢ كيلو", {"كافيه ريش"}, "ar") == "المكان ده على بعد ٢ كيلو"
    assert ps.scrub("روح لكافيه ريش", {"كافيه ريش"}, "ar") == "روح المكان ده"          # proclitic
    # whole words only: a name inside another word is left alone
    assert ps.scrub("Zoobaland is closed", {"Zooba"}, "en") == "Zoobaland is closed"
    # the leading segment and the parts around an abbreviation
    assert ps.scrub("Carrefour is ahead", {"Carrefour - Maadi City Center"}, "en") == "that place is ahead"
    assert ps.scrub("Hamdy Clinic is open", {"Dr. Hamdy Clinic"}, "en") == "that place is open"
    # a stop-listed name is never scrubbed
    assert ps.scrub("In total there are 3 cameras", {"Total"}, "en") == "In total there are 3 cameras"


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
        [("delta", "I found two. Café Riche is the first one, 2 km ahead. Zooba is the second, "
                   "near the Ring Road.")],
    ]
    lines, calls = asyncio.run(run_turn(FakeReq("find me a cafe on the way",
                                                ctx={"dest_name": "Cloud 9 Mall"}), script, monkeypatch))
    spoken = deltas(lines)
    assert "Café Riche" not in spoken and "Zooba" not in spoken
    assert "2 km ahead" in spoken and "Ring Road" in spoken
    sent = json.dumps(calls[-1]["messages"], ensure_ascii=False)
    assert "Café Riche" in sent and '"rating"' not in sent
    assert "SPEAKING PLACES" in sent
    assert [l for l in lines if l["t"] == "action"][0]["action"]["places"][0]["rating"] == 4.6


def test_traffic_speech_keeps_roads_areas_and_the_destination(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)
    script = [[("delta", "Heavy traffic on the Ring Road near Maadi; you'll reach Cloud 9 Mall in 12 minutes.")]]
    ctx = {"dest_name": "Cloud 9 Mall", "alternatives": [{"via": "Ring Road", "leg_to": "Maadi"}]}
    lines, _ = asyncio.run(run_turn(FakeReq("how is traffic", ctx=ctx), script, monkeypatch))
    assert "Ring Road near Maadi" in deltas(lines)
    assert "Cloud 9 Mall" in deltas(lines)


def test_a_not_found_reply_keeps_what_the_user_asked_for(monkeypatch):
    monkeypatch.delenv("COPILOT_SPEAK_PLACE_CONTENT", raising=False)

    async def fake_tool(name, args, ctx):
        return {"found": False, "requested": "gas station", "note": "Nothing nearby."}, None
    monkeypatch.setattr(v2, "execute_tool_v2", fake_tool)
    script = [[("tool_calls", [{"id": "c1", "name": "add_stop", "args": json.dumps({"query": "gas station"})}])],
              [("delta", "I couldn't find a gas station nearby.")]]
    lines, _ = asyncio.run(run_turn(FakeReq("add a gas station"), script, monkeypatch))
    assert "gas station" in deltas(lines)


def test_the_egyptian_form_of_a_name_is_scrubbed_too():
    from api.copilot_egy import masri
    assert ps.scrub(masri("روح بلازا 26 على طول"), {"بلازا 26"}, "ar") == "روح المكان ده على طول"
    assert ps.scrub(masri("روح Cilantro Maadi"), {"Cilantro Maadi"}, "ar") == "روح المكان ده"


def test_round3_lam_alif_numbers_generic_phrases_one_pass():
    from api.copilot_egy import masri
    # ل + ال: «للعزبي»
    assert ps.scrub("روح للعزبي", {"العزبي"}, "ar") == "روح المكان ده"
    assert ps.scrub("هوديك وللعزبي", {"العزبي"}, "ar") == "هوديك المكان ده"
    # a number-only variant never scrubs an ETA
    assert ps.scrub(masri("هتوصل بعد 12 دقيقة"), {"Bldg. 12"}, "ar") == masri("هتوصل بعد 12 دقيقة")
    assert ps.scrub("arrive in 5 minutes", {"Gate No. 5"}, "en") == "arrive in 5 minutes"
    # generic phrases (and Egyptian spellings) are never names
    assert ps.google_names("find_places", {"places": [{"name": "Gas Station"}, {"name": "محطة بنزين"},
                                                       {"name": "صيدليه"}]}) == set()
    assert ps.scrub("I found a gas station 2 km ahead", {"Gas Station - Wataniya"}, "en") \
        == "I found a gas station 2 km ahead"
    # one pass: the placeholder is never scrubbed again
    assert ps.scrub("The Place is near", {"The Place", "Place"}, "en") == "that place is near"


def test_the_users_own_words_in_a_not_found_result_are_not_collected():
    assert ps.google_names("call_place", {"found": False, "requested": "ماما"}) == set()


def test_round4_tatweel_spelling_variants_abbreviations_placeholders_lam():
    assert ps.scrub("روح لـCarrefour Maadi", {"Carrefour Maadi"}, "ar") == "روح المكان ده"
    assert ps.scrub("عايز تروح لـZooba؟", {"Zooba"}, "ar") == "عايز تروح المكان ده؟"
    assert ps.scrub("الـCity Stars زحمة", {"City Stars"}, "ar") == "المكان ده زحمة"
    assert ps.scrub("روح صيدليه العزبي", {"صيدلية العزبي"}, "ar") == "روح المكان ده"
    assert ps.scrub("Dr. Mohamed's clinic is 2 km ahead", {"Dr. Hamdy Clinic"}, "en") \
        == "Dr. Mohamed's clinic is 2 km ahead"
    assert ps.scrub("روح للبان", {"اللبان"}, "ar") == "روح المكان ده"
    # the placeholder is never scrubbed a second time (the emitter runs the scrub twice)
    once = ps.scrub("Go to Place now", {"Place"}, "en")
    assert ps.scrub(once, {"Place"}, "en") == once
    assert ps.google_names("find_places", {"places": [{"name": "Car Wash"}, {"name": "سوبر ماركت"}]}) == set()
