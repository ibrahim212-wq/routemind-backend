# -*- coding: utf-8 -*-
"""
Sekka — the audio path, the language contract on it, and the co-driver behaviours (api/copilot_stt.py,
api/copilot_tts.py, api/copilot_kb.py, api/copilot_weather.py, stream_voice / stream_v2 in api/copilot_v2.py,
the /copilot/voice · /copilot/speak · /copilot/caps endpoints).

THE LANGUAGE PROOF (why the bug is closed, not patched):
  • the transcription request carries NO language and NO conversation text — the language decision is the
    audio's (test_transcription_request_is_language_agnostic);
  • a transcript from that path is never refused by the wrong-mic guard: English after Arabic, Arabic after
    English, one-word answers, low confidence — the reply language is the language just spoken
    (test_reply_language_is_the_language_spoken, parametrized over both directions × short/long/mixed);
  • a fast-path answer that cannot pass the language gate falls through to the model instead of ending the
    turn in silence; "repeat" in the other language than the navigation voice is answered in the driver's.

Run:  python -m pytest tests/test_copilot_sekka.py -q
"""

import asyncio
import io
import json
import struct

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import copilot as base
from api import copilot_stt as stt
from api import copilot_tts as tts
from api import copilot_v2 as v2
from api import usage_limits as ul
from api.copilot_kb import block as kb_block, lookup as kb_lookup
from api.copilot_lang import reply_lang_ok, resolve_language
from api.copilot_weather import summarize


# ── helpers ──────────────────────────────────────────────────────────────────
def wav(seconds: float, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    data = b"\x00\x00" * n
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)


class VoiceReq:
    def __init__(self, prev_lang=None, app_lang="en", history=None, ctx=None, follow_up=False,
                 pending_action=None):
        self.messages = list(history or [])
        self.context = ctx or {}
        self.app_lang = app_lang
        self.prev_lang = prev_lang
        self.pending_action = pending_action
        self.follow_up = follow_up
        self.limit_key = None


def scripted(script):
    calls = []

    async def fake(messages, with_tools, tools=None):
        calls.append({"messages": [dict(m) for m in messages], "with_tools": with_tools})
        events = script[min(len(calls) - 1, len(script) - 1)]
        for ev in events:
            yield ev
        if not events or events[-1][0] not in ("end", "tool_calls"):
            yield ("end", None)
    return fake, calls


def run_voice(monkeypatch, req, heard_text, conf=0.9, script=None, error=None, over=False):
    async def fake_transcribe(audio, filename="speech.wav", content_type="audio/wav", prev_lang=None):
        return stt.SttResult(heard_text, conf, 2.0, error=error)
    monkeypatch.setattr(stt, "transcribe", fake_transcribe)
    fake, calls = scripted(script or [[("delta", "Okay.")]])
    monkeypatch.setattr(base, "_stream_chat", fake)
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")

    async def go():
        return [json.loads(l) async for l in v2.stream_voice(req, wav(2.0), over_budget=over)]
    return asyncio.run(go()), calls


async def _collect(gen):
    return [json.loads(l) async for l in gen]


def said(lines):
    return "".join(l["text"] for l in lines if l["t"] == "delta").strip()


def done(lines):
    return [l for l in lines if l["t"] == "done"][-1]


# ── the clip ─────────────────────────────────────────────────────────────────
def test_wav_duration_reads_the_header():
    assert stt.wav_duration_s(wav(2.5)) == pytest.approx(2.5, abs=1e-3)
    assert stt.wav_duration_s(b"not a wav at all, just bytes" * 3) is None
    # a streaming writer's 0xFFFFFFFF data size → what actually arrived
    w = bytearray(wav(1.0))
    w[40:44] = struct.pack("<I", 0xFFFFFFFF)
    assert stt.wav_duration_s(bytes(w)) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("text,conf,phantom", [
    ("Thanks for watching!", 0.95, True),
    ("اشتركوا في القناة", 0.9, True),
    ("ترجمة نانسي قنقر", 0.9, True),
    ("you you you", 0.9, True),
    ("Thank you.", 0.30, True),          # unsure courtesy phrase in noise
    ("Thank you.", 0.92, False),         # a real thank-you
    ("شكرا", 0.9, False),
    ("", None, True),
    ("عايز أروح الدائري", 0.8, False),
])
def test_phantom_filter(text, conf, phantom):
    assert stt.is_phantom(text, conf) is phantom


def test_confidence_from_logprobs():
    assert stt.confidence_from_logprobs([{"logprob": 0.0}, {"logprob": 0.0}]) == 1.0
    assert stt.confidence_from_logprobs([{"logprob": -0.6931}]) == pytest.approx(0.5, abs=1e-3)
    assert stt.confidence_from_logprobs(None) is None


def test_transcription_is_a_bilingual_jury(monkeypatch):
    """The root-cause property: the clip is heard three ways in parallel — Arabic with an ARABIC prompt, English
    with an ENGLISH prompt (each prompt matches the language it forces, per the provider's guidance), and once
    with no language and NO prompt — and never with the conversation's own text."""
    seen = []

    class R:
        status_code = 200
        text = ""

        def __init__(self, data):
            self.data = data

        def json(self):
            if self.data.get("language") == "ar":
                return {"text": "هاو لونج إز ليفت", "logprobs": [{"logprob": -0.9}]}
            return {"text": "How long is left?", "logprobs": [{"logprob": -0.05}]}

    class C:
        async def post(self, url, headers=None, data=None, files=None):
            seen.append(dict(data))
            assert files["file"][0] == "speech.wav"
            return R(data)

    monkeypatch.setattr(stt, "_client", C())
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    monkeypatch.setattr(stt, "JURY", True)
    r = asyncio.run(stt.transcribe(wav(1.5), prev_lang="ar"))
    assert r.text == "How long is left?" and r.lang == "en" and r.confidence > 0.9
    by = {d.get("language", "auto"): d for d in seen}
    assert set(by) == {"ar", "en", "auto"}
    assert by["ar"]["prompt"] == stt.AR_PROMPT and by["en"]["prompt"] == stt.EN_PROMPT
    assert "prompt" not in by["auto"]
    assert all("؀" <= c <= "ۿ" or not c.isalpha() for c in stt.AR_PROMPT.replace(" ", ""))
    assert not any("؀" <= c <= "ۿ" for c in stt.EN_PROMPT)
    assert all(d["model"] == stt.STT_MODEL for d in seen)


def test_too_short_clip_is_not_sent(monkeypatch):
    class C:
        async def post(self, *a, **k):
            raise AssertionError("a sub-syllable clip must never cost a request")
    monkeypatch.setattr(stt, "_client", C())
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    r = asyncio.run(stt.transcribe(wav(0.1)))
    assert r.text == "" and r.error is None


# ── the language contract on the audio path ──────────────────────────────────
EN_REPLY = [[("delta", "About twelve minutes to go.")]]
AR_REPLY = [[("delta", "فاضل حوالي اتناشر دقيقة.")]]


@pytest.mark.parametrize("prev,heard,expected", [
    ("ar", "How long is left?", "en"),                  # English after an Arabic conversation
    ("ar", "Yes.", "en"),                               # one English word
    ("ar", "Where's the nearest gas station?", "en"),
    ("en", "فاضل كام؟", "ar"),                          # Arabic after an English conversation
    ("en", "لأ", "ar"),                                  # one Arabic word
    ("en", "ايوه", "ar"),
    ("en", "خدني عالـ Ring Road", "ar"),                # Egyptian sentence carrying an English road name
    ("ar", "take me to التجمع الخامس", "en"),           # English sentence carrying an Arabic place
    (None, "How far?", "en"),
    (None, "هوصل امتى؟", "ar"),
])
def test_reply_language_is_the_language_spoken(monkeypatch, prev, heard, expected):
    lines, _ = run_voice(monkeypatch, VoiceReq(prev_lang=prev, app_lang="ar" if prev == "en" else "en"),
                         heard, conf=0.9, script=EN_REPLY if expected == "en" else AR_REPLY)
    assert lines[0] == {"t": "meta", "lang": expected, "v": 2}
    assert lines[1]["t"] == "heard" and lines[1]["text"] == heard
    assert said(lines) and reply_lang_ok(said(lines), expected)
    assert done(lines)["lang"] == expected


def test_low_confidence_audio_switch_is_not_refused(monkeypatch):
    """The wrong-mic guard belongs to monolingual recognizers only: the audio path has no wrong mic."""
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar"), "What's the traffic like ahead?",
                             conf=0.40, script=EN_REPLY)
    assert lines[0]["lang"] == "en"
    assert calls, "the turn reached the model — it was not turned into an 'unclear' question"
    # the same text from an Arabic on-device recognizer at that confidence is still guarded
    r = resolve_language("What's the traffic like ahead?", prev_lang="ar", stt_lang="en", stt_confidence=0.40)
    assert r.lang == "ar" and r.source == "guard"


def test_very_low_confidence_audio_is_flagged_unreliable():
    r = resolve_language("go to the airport", prev_lang="ar", stt_confidence=0.2, stt_source="audio")
    assert r.lang == "en" and r.unreliable


def test_history_arrives_without_the_new_turn_and_gets_it(monkeypatch):
    hist = [{"role": "user", "content": "عايز أسرع طريق"}, {"role": "assistant", "content": "انت عليه."}]
    _, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar", history=hist), "okay and how long left then",
                         script=EN_REPLY)
    user_msgs = [m for m in calls[0]["messages"] if m["role"] == "user"]
    assert user_msgs[0]["content"] == "عايز أسرع طريق"
    assert user_msgs[-1]["content"].startswith("okay and how long left then")


def test_nothing_heard_asks_once_then_follow_up_silence_ends(monkeypatch):
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar"), "")
    assert not calls
    assert said(lines) and reply_lang_ok(said(lines), "ar")
    assert done(lines)["open_mic"] is True
    lines, _ = run_voice(monkeypatch, VoiceReq(prev_lang="ar", follow_up=True), "")
    assert said(lines) == "" and done(lines)["end"] is True and done(lines)["open_mic"] is False


def test_unsure_speech_in_a_follow_up_window_is_not_answered(monkeypatch):
    """The radio or a passenger in the few seconds the mic stays open after a reply."""
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="en", follow_up=True),
                             "and the weather tomorrow in Alexandria", conf=0.42)
    assert not calls and said(lines) == "" and done(lines)["end"] is True
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="en", follow_up=True),
                             "and how far is it", conf=0.9, script=EN_REPLY)
    assert calls and said(lines)


def test_ears_failure_is_spoken_in_the_conversation_language(monkeypatch):
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar"), "", error="transport")
    err = [l for l in lines if l["t"] == "error"][0]
    assert err["code"] == "stt_failed" and reply_lang_ok(err["spoken"], "ar") and not calls


def test_voice_budget_exhausted(monkeypatch):
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="en"), "hello", over=True)
    err = [l for l in lines if l["t"] == "error"][0]
    assert err["code"] == "voice_limit" and not calls


# ── never silent ─────────────────────────────────────────────────────────────
def test_fastpath_that_fails_the_gate_falls_through_to_the_model(monkeypatch):
    async def no_translation(text, lang):
        return None
    monkeypatch.setattr(v2, "_translate", no_translation)
    ctx = {"current_road": "26th of July Corridor Westbound Express Lanes", "remaining_km": 10}
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar", ctx=ctx),
                             "انا ماشي على طريق ايه دلوقتي",
                             script=[[("delta", "انت ماشي على محور ستة وعشرين يوليو.")]])
    assert calls, "the model answered"
    assert said(lines) and reply_lang_ok(said(lines), "ar")


@pytest.mark.parametrize("prev,heard,nav_lang,maneuver,reply", [
    ("ar", "repeat that", "ar", "خليك يمين", "In about four hundred meters, keep right."),
    ("en", "قول تاني", "en", "Turn left onto Tahrir Street", "بعد حوالي تلتمية متر، لف شمال على شارع التحرير."),
])
def test_repeat_in_the_other_language_than_navigation_is_answered_in_the_drivers(monkeypatch, prev, heard,
                                                                               nav_lang, maneuver, reply):
    """Never a guidance replay in a language the driver did not speak, never a cross-script splice."""
    ctx = {"nav_lang": nav_lang, "next_maneuver": maneuver, "next_maneuver_m": 400}
    lang = "en" if nav_lang == "ar" else "ar"
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang=prev, ctx=ctx), heard, script=[[("delta", reply)]])
    assert lines[0]["lang"] == lang
    assert not [l for l in lines if l["t"] == "action"], "no replay of the other-language guidance line"
    assert calls and "do NOT call repeat_instruction" in calls[0]["messages"][-1]["content"]
    assert said(lines) == reply and reply_lang_ok(said(lines), lang)


def test_repeat_is_spoken_from_the_template_when_the_trip_text_is_in_the_drivers_script(monkeypatch):
    ctx = {"nav_lang": "en", "next_maneuver": "خليك يمين", "next_maneuver_m": 400}   # e.g. an Arabic road SDK
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar", ctx=ctx), "قول تاني")
    assert not calls and lines[0]["lang"] == "ar" and "خليك يمين" in said(lines)


@pytest.mark.parametrize("heard,lang,ctx", [
    ("what's my next turn", "en", {"next_maneuver": "خليك يمين", "next_maneuver_m": 300}),
    ("اللفة الجاية فين", "ar", {"next_maneuver": "Turn left onto Tahrir Street", "next_maneuver_m": 300}),
])
def test_fastpath_never_splices_the_other_script(monkeypatch, heard, lang, ctx):
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang=lang, ctx=ctx), heard,
                             script=EN_REPLY if lang == "en" else AR_REPLY)
    assert calls, "the model answers instead of a mixed-script template"
    text = said(lines)
    assert not any("\u0600" <= c <= "\u06ff" for c in text) if lang == "en" else not any(c.isascii() and c.isalpha() for c in text)


def test_same_language_repeat_replays_the_guidance(monkeypatch):
    ctx = {"nav_lang": "en", "next_maneuver": "Turn left", "next_maneuver_m": 300}
    lines, _ = run_voice(monkeypatch, VoiceReq(prev_lang="en", ctx=ctx), "repeat that")
    assert [l["action"]["type"] for l in lines if l["t"] == "action"] == ["repeat_instruction"]


# ── conversation: knowing when it is over ────────────────────────────────────
@pytest.mark.parametrize("heard,lang", [("خلاص كده", "ar"), ("that's all", "en"), ("no thanks", "en"),
                                        ("شكرا", "ar"), ("thank you", "en"), ("مع السلامة", "ar")])
def test_sign_offs_end_the_conversation_without_the_model(monkeypatch, heard, lang):
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang=lang), heard)
    assert not calls
    assert reply_lang_ok(said(lines), lang)
    assert done(lines)["end"] is True and done(lines)["open_mic"] is False


def test_a_question_reopens_the_mic(monkeypatch):
    lines, _ = run_voice(monkeypatch, VoiceReq(prev_lang="en"), "add a stop",
                         script=[[("delta", "Sure — gas, food or pharmacy?")]])
    assert done(lines)["open_mic"] is True and done(lines)["end"] is False


def test_closer_while_an_action_is_pending_goes_to_the_model(monkeypatch):
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar", pending_action={"type": "add_stop"}),
                             "خلاص", script=[[("delta", "تمام، مش هضيفها.")]])
    assert calls


@pytest.mark.parametrize("heard,answer", [("أيوه", "yes"), ("yes please", "yes"), ("تمام", "yes"),
                                          ("لأ", "no"), ("no", "no"), ("لا مش عايز", "no")])
def test_short_answer_to_a_pending_action_is_the_phones_to_execute(monkeypatch, heard, answer):
    lines, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar", pending_action={"type": "add_stop"}), heard)
    assert not calls
    acts = [l["action"] for l in lines if l["t"] == "action"]
    assert acts == [{"type": "confirm_pending", "answer": answer, "requires_confirm": False}]


@pytest.mark.parametrize("heard", ["لا التانية", "no the second one please", "yes but not that one"])
def test_a_refinement_is_the_models(monkeypatch, heard):
    _, calls = run_voice(monkeypatch, VoiceReq(prev_lang="ar", pending_action={"type": "add_stop"}), heard,
                         script=[[("delta", "تمام.")]])
    assert calls


# ── Sekka on the car screen (no phone navigation behind it) ─────────────────────
CAR_CAN = ["find_places", "place_details", "traffic_check", "weather", "switch_route", "show_alternatives",
           "show_overview", "set_guidance_voice", "set_view_mode", "set_voice_volume", "repeat_instruction",
           "report_incident", "remind_in", "remind_before_arrival", "emergency_call", "call_place",
           "cancel_navigation"]


def test_the_car_is_offered_only_what_it_can_do(monkeypatch):
    seen = []

    async def fake(messages, with_tools, tools=None):
        seen.append(tools)
        yield ("delta", "Okay.")
        yield ("end", None)
    monkeypatch.setattr(base, "_stream_chat", fake)
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    req = VoiceReq(prev_lang="en", ctx={"surface": "car", "can": CAR_CAN},
                   history=[{"role": "user", "content": "What is the weather like at the destination?"}])
    asyncio.run(_collect(v2.stream_v2(req)))
    names = {t["function"]["name"] for t in seen[0]}
    # what the car can execute, plus the tools the SERVER answers from data (no executor on any surface)
    assert names == (set(CAR_CAN) | v2.INFO_TOOLS) & _tool_names()
    assert {"plan_departure", "prayer_times", "eta_to"} <= names
    assert "add_stop" not in names and "route_options" not in names
    # the phone (no `can`) keeps every tool
    assert len(v2.tools_for({})) == len(v2.TOOLS_V2)


def test_a_tool_the_car_cannot_run_is_refused_not_executed(monkeypatch):
    script = [[("tool_calls", [{"id": "c1", "name": "add_stop", "args": json.dumps({"query": "gas station"})}])],
              [("delta", "That one is on the phone.")]]
    fake, calls = scripted(script)
    monkeypatch.setattr(base, "_stream_chat", fake)
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    req = VoiceReq(prev_lang="en", ctx={"surface": "car", "can": CAR_CAN},
                   history=[{"role": "user", "content": "Add a gas station on the way"}])
    lines = asyncio.run(_collect(v2.stream_v2(req)))
    assert not [l for l in lines if l["t"] == "action"]
    tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and json.loads(tool_msgs[0]["content"]).get("unavailable") is True


def test_a_fast_action_the_car_cannot_run_goes_to_the_model(monkeypatch):
    fake, calls = scripted([[("delta", "Okay.")]])
    monkeypatch.setattr(base, "_stream_chat", fake)
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    req = VoiceReq(prev_lang="en", ctx={"can": ["weather"]},
                   history=[{"role": "user", "content": "mute"}])
    lines = asyncio.run(_collect(v2.stream_v2(req)))
    assert not [l for l in lines if l["t"] == "action"] and calls


def test_the_car_surface_is_named_in_the_trip_block():
    assert "CAR screen" in v2._format_context_v2({"surface": "car", "speed_kmh": 40})
    assert "CAR screen" not in v2._format_context_v2({"speed_kmh": 40})


# ── identity + new abilities ─────────────────────────────────────────────────
def test_identity_is_sekka():
    assert "Sekka (سِكّة)" in v2.SYSTEM_V2
    assert "RouteMind Copilot" not in v2.SYSTEM_V2


def _tool_names():
    return {t["function"]["name"] for t in v2.TOOLS_V2}


def test_new_tools_are_offered():
    assert {"weather", "remind_in", "emergency_call"} <= _tool_names()


@pytest.mark.parametrize("service,number", [("police", "122"), ("ambulance", "123"), ("fire", "180")])
def test_emergency_call_previews_a_dial_and_needs_a_yes(service, number):
    res, action = asyncio.run(v2.execute_tool_v2("emergency_call", {"service": service}, {"_turn_lang": "ar"}))
    assert action["type"] == "dial" and action["number"] == number
    assert action["requires_confirm"] is True and action["emergency"] is True
    assert any("؀" <= c <= "ۿ" for c in action["place"])


def test_remind_in():
    res, action = asyncio.run(v2.execute_tool_v2("remind_in", {"minutes": 20, "about": "تكلم أحمد"}, {}))
    assert action == {"type": "remind_in", "minutes": 20, "about": "تكلم أحمد",
                      "requires_confirm": False, "commit": "done"}
    _, cancel = asyncio.run(v2.execute_tool_v2("remind_in", {"minutes": 0}, {}))
    assert cancel["minutes"] == 0
    _, capped = asyncio.run(v2.execute_tool_v2("remind_in", {"minutes": 9999, "about": "x"}, {}))
    assert capped["minutes"] == 240


def test_weather_summary_flags_what_changes_a_drive():
    body = {"current": {"temperature_2m": 14.4, "apparent_temperature": 13, "weather_code": 45,
                        "wind_speed_10m": 6, "wind_gusts_10m": 12, "visibility": 400},
            "hourly": {"weather_code": [45, 3, 1], "precipitation_probability": [0, 5, 0]}}
    s = summarize(body, "here")
    assert s["conditions"] == "fog" and s["visibility_km"] == 0.4 and s["temp_c"] == 14
    assert any("LOW VISIBILITY" in f for f in s["driving_flags"])
    hot = summarize({"current": {"temperature_2m": 42.2, "weather_code": 0}}, "destination")
    assert hot["where"] == "destination" and "extreme heat" in hot["driving_flags"]


def test_weather_without_a_position_says_so():
    res = asyncio.run(__import__("api.copilot_weather", fromlist=["weather"]).weather({}, "destination"))
    assert res["available"] is False


@pytest.mark.parametrize("q,key", [("is there fog on the desert road", "fog"), ("الكاوتش فرقع", "breakdown"),
                                   ("امتي الطريق يبقى فاضي", "rush"), ("I need an ambulance", "emergency"),
                                   ("فيه كارتة على الطريق ده؟", "tolls"), ("رمضان والفطار", "ramadan")])
def test_knowledge_is_retrieved_only_when_asked(q, key):
    assert key in [e.key for e in kb_lookup(q)]


@pytest.mark.parametrize("q", ["navigate to the mall", "take me to town", "how long left", "الجامعة فين"])
def test_no_knowledge_tokens_for_plain_requests(q):
    assert kb_block(q) == ""


def test_knowledge_reaches_the_model_after_the_static_prefix(monkeypatch):
    _, calls = run_voice(monkeypatch, VoiceReq(prev_lang="en"), "is there fog on the desert road this morning",
                         script=EN_REPLY)
    msgs = calls[0]["messages"]
    assert msgs[0]["content"] == v2.SYSTEM_V2                      # cache-stable prefix
    assert "[Local knowledge" in msgs[-1]["content"] and "الشبورة" in msgs[-1]["content"]


# ── endpoints ────────────────────────────────────────────────────────────────
@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(base.router, prefix="/api")
    monkeypatch.setattr(base, "OPENAI_KEY", "sk-test")
    monkeypatch.setattr(ul, "VOICE_SECONDS", ul.DailyCounter())
    monkeypatch.setattr(ul, "TTS_CHARS", ul.DailyCounter())
    return TestClient(app)


def test_caps(client):
    c = client.get("/api/copilot/caps").json()
    assert c["voice"] is True and c["speak"] is True and c["name"] == {"ar": "سِكّة", "en": "Sekka"}


def test_voice_endpoint_streams_meta_heard_reply(client, monkeypatch):
    async def fake_transcribe(audio, filename="speech.wav", content_type="audio/wav", prev_lang=None):
        assert stt.wav_duration_s(audio) == pytest.approx(1.5, abs=1e-3)
        return stt.SttResult("How long is left?", 0.93, 1.5)
    monkeypatch.setattr(stt, "transcribe", fake_transcribe)
    fake, _ = scripted(EN_REPLY)
    monkeypatch.setattr(base, "_stream_chat", fake)
    payload = {"messages": [{"role": "user", "content": "عايز أسرع طريق"},
                            {"role": "assistant", "content": "انت عليه."}],
               "prev_lang": "ar", "app_lang": "ar", "context": {}, "device_id": "device-test-0001"}
    r = client.post("/api/copilot/voice", files={"audio": ("speech.wav", wav(1.5), "audio/wav")},
                    data={"payload": json.dumps(payload)})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    assert [l["t"] for l in lines[:2]] == ["meta", "heard"]
    assert lines[0]["lang"] == "en" and said(lines) == "About twelve minutes to go."


def test_voice_endpoint_rejects_an_overlong_clip(client):
    r = client.post("/api/copilot/voice", files={"audio": ("speech.wav", wav(30.0), "audio/wav")},
                    data={"payload": "{}"})
    assert r.status_code == 413


def test_voice_budget_is_charged_in_seconds(client, monkeypatch):
    monkeypatch.setenv("COPILOT_VOICE_SECONDS_PER_DAY", "3")
    seen = []

    async def fake_transcribe(audio, filename="speech.wav", content_type="audio/wav", prev_lang=None):
        seen.append(1)
        return stt.SttResult("thanks", 0.95, 2.0)
    monkeypatch.setattr(stt, "transcribe", fake_transcribe)
    body = {"payload": json.dumps({"device_id": "device-test-0002", "prev_lang": "en"})}
    first = client.post("/api/copilot/voice", files={"audio": ("s.wav", wav(2.0), "audio/wav")}, data=body)
    second = client.post("/api/copilot/voice", files={"audio": ("s.wav", wav(2.0), "audio/wav")}, data=body)
    third = client.post("/api/copilot/voice", files={"audio": ("s.wav", wav(2.0), "audio/wav")}, data=body)
    assert '"voice_limit"' not in first.text and '"voice_limit"' not in second.text
    assert '"voice_limit"' in third.text and len(seen) == 2


def test_speak_endpoint_and_its_budget(client, monkeypatch):
    monkeypatch.setattr(tts, "CACHE", tts._Lru(1 << 20))
    sent = []

    class R:
        status_code = 200
        content = b"ID3fake-mp3"
        text = ""

    class C:
        async def post(self, url, json=None, headers=None):
            sent.append(json)
            return R()
    monkeypatch.setattr(tts, "_client", C())
    monkeypatch.setenv("COPILOT_TTS_CHARS_PER_DAY", "30")
    r = client.post("/api/copilot/speak", json={"text": "قدامك زحمة بسيطة.", "lang": "ar",
                                                "device_id": "device-test-0003"})
    assert r.status_code == 200 and r.content == b"ID3fake-mp3" and r.headers["content-type"] == "audio/mpeg"
    assert "Cairene" in sent[0]["instructions"] and sent[0]["response_format"] == "mp3"
    # the same line again is a cache hit: no provider call, no budget
    r2 = client.post("/api/copilot/speak", json={"text": "قدامك زحمة بسيطة.", "lang": "ar",
                                                 "device_id": "device-test-0003"})
    assert r2.status_code == 200 and len(sent) == 1
    long_line = "x" * 25
    ok = client.post("/api/copilot/speak", json={"text": long_line, "lang": "en", "device_id": "device-test-0003"})
    over = client.post("/api/copilot/speak", json={"text": long_line + "y", "lang": "en",
                                                   "device_id": "device-test-0003"})
    assert ok.status_code == 200 and over.status_code == 429


def test_converse_endpoint_ignores_a_client_claimed_audio_source(client, monkeypatch):
    fake, calls = scripted(AR_REPLY)
    monkeypatch.setattr(base, "_stream_chat", fake)
    body = {"messages": [{"role": "user", "content": "how are you the road"}], "prev_lang": "ar",
            "stt_lang": "en", "stt_confidence": 0.3, "stt_source": "audio"}
    r = client.post("/api/copilot/converse", json=body)
    first = json.loads(r.text.splitlines()[0])
    assert first["lang"] == "ar"            # the wrong-mic guard still protects the on-device path
