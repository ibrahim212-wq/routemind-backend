# -*- coding: utf-8 -*-
"""
The CopilotV2 LANGUAGE HARNESS — Phase-1 mandatory verification.

Requirement: user speaks/types Arabic → assistant replies in Arabic; English →
English; Arabizi counts as Arabic; mixed resolves to the dominant intent
language; sticky across garbled turns; every error string localized. 100%.

The harness drives the SAME functions the live stream runs:
  resolve_language()  — the single per-turn decision
  reply_lang_ok()     — the deterministic output gate (nothing mismatched can
                        be emitted past it, by construction of _gated_pass)
  copilot_strings     — the backend catalog (client catalogs mirrored by
                        tools/check_copilot_parity.py)

Run:  python -m pytest tests/test_copilot_language.py -q
"""

import re

import pytest

from api.copilot_lang import (resolve_language, reply_lang_ok, is_arabizi,
                              split_sentences, speak_minutes, speak_distance,
                              explicit_language_request, script_counts)
from api import copilot_strings


# ═══════════════════════════════════════════════════════════════════════════
# 1. resolve_language — the single authoritative decision
#    Each case: (input text, prev_lang, app_lang, expected lang)
# ═══════════════════════════════════════════════════════════════════════════

PURE_ARABIC = [
    ("كام رادار قدامي؟", None, "en", "ar"),
    ("الزحمة دي عادية الوقت ده ولا لأ؟", None, "en", "ar"),
    ("عايز أسرع طريق", None, "en", "ar"),
    ("فاضل كام على ما نوصل؟", None, "en", "ar"),
    ("وريني الطرق البديلة", None, "en", "ar"),
    ("ضيفلي أقرب بنزينة", None, "en", "ar"),
    ("خد بالك في كاميرا قدام؟", None, "en", "ar"),
    ("اقفل الصوت", None, "en", "ar"),
    ("انا ماشي على طريق ايه دلوقتي", None, "en", "ar"),
    ("هوصل الساعة كام؟", None, "en", "ar"),
    ("في زحمة على الدائري؟", None, "en", "ar"),
    ("شيل الوقفة اللي ضفتها", None, "en", "ar"),
    ("بلاش الطريق ده خالص", None, "en", "ar"),
    ("عايز اروح مدينتي بدل التجمع", None, "en", "ar"),
    ("افتكرلي أركن فين", None, "en", "ar"),
    ("خليك معايا شوية", None, "en", "ar"),
    ("ايه رأيك في ماتش الأهلي امبارح", None, "en", "ar"),
    ("سرعتي كام دلوقتي", None, "en", "ar"),
    ("فيه صيدلية جنبي؟", None, "en", "ar"),
    ("ابعت وصولي لمراتي", None, "en", "ar"),
]

PURE_ENGLISH = [
    ("how many cameras ahead?", None, "ar", "en"),
    ("is this traffic normal for this time?", None, "ar", "en"),
    ("take the fastest route", None, "ar", "en"),
    ("how long until we arrive?", None, "ar", "en"),
    ("show me the alternative routes", None, "ar", "en"),
    ("add the nearest gas station", None, "ar", "en"),
    ("is there a camera coming up?", None, "ar", "en"),
    ("mute the voice", None, "ar", "en"),
    ("what road am I on right now?", None, "ar", "en"),
    ("what time will I get there?", None, "ar", "en"),
    ("any traffic on the ring road?", None, "ar", "en"),
    ("remove the stop I added", None, "ar", "en"),
    ("avoid this road entirely", None, "ar", "en"),
    ("I want to go to Madinaty instead", None, "ar", "en"),
    ("remind me where I parked", None, "ar", "en"),
    ("stay with me for a bit", None, "ar", "en"),
    ("what did you think of the match yesterday", None, "ar", "en"),
    ("take the 2nd one", None, "ar", "en"),          # ordinal ≠ Arabizi
    ("wake me up at 8am near the exit", None, "ar", "en"),
    ("share my ETA with my wife", None, "ar", "en"),
]

ARABIZI = [  # Latin-script Arabic MUST resolve to Arabic
    ("ana 3ayez asra3 tare2", None, "en", "ar"),
    ("fein a2rab banzeena", None, "en", "ar"),
    ("el za7ma di 3adia?", None, "en", "ar"),
    ("warini el turo2 el tanya", None, "en", "ar"),
    ("kam radar 2odamy", None, "en", "ar"),
    ("ana gay men el tagamo3", None, "en", "ar"),
    ("5od balak fe kamera odam", None, "en", "ar"),
    ("mesh 3ayez el tari2 dah", None, "en", "ar"),
    ("yalla bina", None, "en", "ar"),
    ("tamam ya basha kamel", "ar", "en", "ar"),
    ("e7na hanewsal emta", None, "en", "ar"),
    ("hatly a2rab saydalia", None, "en", "ar"),
    ("el mozza3 fein", "ar", "en", "ar"),
    ("ma3lesh 3edt el mokalma", None, "en", "ar"),
    ("sheel el wa2fa elly fatet", None, "en", "ar"),
]

MIXED = [  # dominant-intent language wins; borrowed tokens don't flip it
    ("خدني عالring road", None, "en", "ar"),
    ("خدني عالring road", "en", "en", "ar"),          # even with English history
    ("add stop على أقرب gas station", None, "en", "en"),
    ("عايز اروح Mall of Egypt", None, "en", "ar"),
    ("قفلي على Master اللي جاية", None, "en", "ar"),
    ("switch to الدائري please", None, "ar", "en"),
    ("هي Cilantro لسه فاتحة؟", None, "en", "ar"),
    ("is المحور faster right now?", None, "ar", "en"),
    ("وريني أقرب Starbucks", None, "en", "ar"),
    ("ok خلاص كمل على طريقك", None, "en", "ar"),
    ("لا McDonald's مش KFC", None, "en", "ar"),
    ("get me to التجمع الخامس", None, "ar", "en"),
]

SINGLE_WORD = [
    ("تمام", None, "en", "ar"),
    ("ايوه", None, "en", "ar"),
    ("لا", None, "en", "ar"),
    ("نعم", None, "en", "ar"),
    ("yes", None, "ar", "en"),
    ("cancel", None, "ar", "en"),
    ("sure", None, "ar", "en"),
    ("stop", None, "ar", "en"),
    # borrowed-only words stay with the conversation
    ("ok", "ar", "en", "ar"),
    ("ok", "en", "ar", "en"),
    ("okay", "ar", "en", "ar"),
]

NOISY_STT = [
    # garbled fragments: stickiness must hold the conversation language
    ("mmm uh the uh", "ar", "ar", "en"),      # real English words → English
    ("ال ال ممم", "en", "en", "ar"),           # Arabic script fragments → Arabic
    ("asdkjh qwerty", "ar", "en", "en"),       # Latin junk, no Arabizi markers
    ("ةةة ييي", "en", "en", "ar"),
    ("hhh", "ar", "en", "en"),
    ("ااا", "en", "en", "ar"),
    ("123 456", "ar", "en", "ar"),             # digits only → sticky prev
    ("123 456", "en", "ar", "en"),
    ("...", "ar", "en", "ar"),                 # punctuation only → sticky prev
    ("؟", "en", "ar", "en"),
]

EMPTY = [
    ("", "ar", "en", "ar"),
    ("   ", "en", "ar", "en"),
    ("", None, "ar", "ar"),
    ("", None, "en", "en"),
    ("", None, "xx", "en"),                    # bad app_lang → safe default
]

EXPLICIT = [
    ("speak arabic please", None, "en", "ar"),
    ("answer in arabic", "en", "en", "ar"),
    ("بالعربي لو سمحت", "en", "en", "ar"),
    ("اتكلم عربي", "en", "en", "ar"),
    ("in english please", "ar", "ar", "en"),
    ("speak english", "ar", "ar", "en"),
    ("بالانجليزي", "ar", "ar", "en"),
    ("كلمني انجليزي من فضلك", "ar", "ar", "en"),
]

ALL_RESOLVE = (PURE_ARABIC + PURE_ENGLISH + ARABIZI + MIXED + SINGLE_WORD
               + NOISY_STT + EMPTY + EXPLICIT)


@pytest.mark.parametrize("text,prev,app,expected", ALL_RESOLVE)
def test_resolve_language(text, prev, app, expected):
    got = resolve_language(text, prev_lang=prev, app_lang=app)
    assert got.lang == expected, (
        f"{text!r} (prev={prev}, app={app}) → {got.lang} ({got.source}), "
        f"expected {expected}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Multi-turn drift — language re-resolved per turn but sticky
# ═══════════════════════════════════════════════════════════════════════════

def _run_conversation(turns):
    prev = None
    out = []
    for text in turns:
        r = resolve_language(text, prev_lang=prev, app_lang="en")
        out.append(r.lang)
        prev = r.lang
    return out


def test_drift_arabic_conversation_survives_garbage():
    langs = _run_conversation([
        "عايز أسرع طريق", "تمام", "ااا ممم", "123", "وبعدين؟"])
    assert langs == ["ar", "ar", "ar", "ar", "ar"]


def test_drift_english_conversation_survives_garbage():
    langs = _run_conversation([
        "fastest route please", "yes", "hhh", "...", "and then?"])
    assert langs == ["en", "en", "en", "en", "en"]


def test_drift_genuine_switch_is_honored():
    langs = _run_conversation([
        "how long left?", "ok", "طب وريني الزحمة", "وبعدين؟"])
    assert langs == ["en", "en", "ar", "ar"]


def test_drift_switch_back():
    langs = _run_conversation([
        "كام رادار قدامي", "ايوه", "now switch to english mode please",
        "how far to the next one?"])
    assert langs[-2:] == ["en", "en"]


def test_drift_borrowed_brand_does_not_flip():
    langs = _run_conversation([
        "ضيفلي وقفة", "Master", "ايوه ضيفها"])
    assert langs == ["ar", "ar", "ar"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. The output validator — the deterministic reply gate
# ═══════════════════════════════════════════════════════════════════════════

AR_REPLIES_OK = [
    "فاضل حوالي عشر دقايق، هتوصل قبل الساعة خمسة.",
    "قدامك رادارين — أقربهم بعد كيلو ونص، والسرعة عنده 80.",
    "أقرب بنزينة Master على بعد 3 كيلومترات، أضيفها؟",
    "El Dahan تقييمه 4.6 من 3000 تقييم وهو فاتح دلوقتي.",
    "اتحولت للدائري — هتكسب حوالي 5 دقايق.",
    "معلش، حصلت مشكلة. جرب تاني.",
    "الزحمة النهارده أتقل من العادي في الوقت ده.",
]
EN_REPLIES_OK = [
    "About ten minutes to go, arriving around 5.",
    "Two cameras ahead — the nearest in about a kilometer and a half.",
    "The nearest Master is 3 kilometers away, want me to add it?",
    "El Dahan is 4.6 with 3,000 reviews and open now.",
    "Switched to the Ring Road — saves you about 5 minutes.",
    "Sorry, something went wrong. Try again.",
    "Traffic is heavier than usual for this hour on المحور.",
]
NAME_ONLY = ["Master.", "Cilantro التجمع", "KFC?", "19011"]

WRONG_FOR_AR = [
    "Sorry, I did not understand that, please repeat.",
    "The nearest gas station is Master, about 3 kilometers ahead.",
    "There is heavy traffic on the Ring Road costing about 7 minutes.",
    "I can only answer in English.",
]
WRONG_FOR_EN = [
    "معلش مش فاهم، ممكن تعيد تاني؟",
    "أقرب بنزينة على بعد 3 كيلو.",
    "في زحمة تقيلة على الدائري هتأخرك 7 دقايق.",
]


@pytest.mark.parametrize("reply", AR_REPLIES_OK + NAME_ONLY)
def test_validator_accepts_valid_arabic_turn(reply):
    assert reply_lang_ok(reply, "ar")


@pytest.mark.parametrize("reply", EN_REPLIES_OK + NAME_ONLY)
def test_validator_accepts_valid_english_turn(reply):
    assert reply_lang_ok(reply, "en")


@pytest.mark.parametrize("reply", WRONG_FOR_AR)
def test_validator_rejects_english_when_arabic_pinned(reply):
    assert not reply_lang_ok(reply, "ar")


@pytest.mark.parametrize("reply", WRONG_FOR_EN)
def test_validator_rejects_arabic_when_english_pinned(reply):
    assert not reply_lang_ok(reply, "en")


def test_validator_empty_and_numeric_pass():
    assert reply_lang_ok("", "ar")
    assert reply_lang_ok("80.", "en")
    assert reply_lang_ok("5:45.", "ar")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Error-path strings — every catalog key localized, scripts pure
#    (client catalogs are proven in sync by tools/check_copilot_parity.py)
# ═══════════════════════════════════════════════════════════════════════════

_AR_RE = re.compile(r"[؀-ۿ]")


def test_backend_catalog_complete_and_script_pure():
    assert copilot_strings.KEYS, "catalog must not be empty"
    for key in copilot_strings.KEYS:
        ar = copilot_strings.t(key, "ar")
        en = copilot_strings.t(key, "en")
        assert ar and en, f"{key} missing a language"
        assert _AR_RE.search(ar), f"{key} 'ar' value has no Arabic script: {ar!r}"
        assert not _AR_RE.search(en), f"{key} 'en' value contains Arabic: {en!r}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Building blocks
# ═══════════════════════════════════════════════════════════════════════════

def test_arabizi_detector_positive():
    for s in ["3ayez", "za7ma", "tare2", "ana 3ayez asra3 tare2",
              "fein el banzeena", "yalla bina", "ana mashy delwa2ty"]:
        assert is_arabizi(s), s


def test_arabizi_detector_negative():
    for s in ["take the 2nd exit", "wake me at 8am", "play mp3", "route 66",
              "the 3rd one on the left", "5 km more", "see you l8r no wait",
              "meet me at 7", "gate b2"]:
        assert not is_arabizi(s), s


def test_explicit_request_detection():
    assert explicit_language_request("يا ريت بالانجليزي") == "en"
    assert explicit_language_request("please speak arabic") == "ar"
    assert explicit_language_request("عادي كمل") is None


def test_script_counts_ignore_arabic_digits():
    ar, en = script_counts("٥ دقايق")
    assert ar == 5 and en == 0


def test_split_sentences_decimal_safe():
    out, rest = split_sentences("about 2.5 km ahead. then turn", force=False)
    assert out == ["about 2.5 km ahead."]
    assert rest == " then turn"
    out2, rest2 = split_sentences(rest, force=True)
    assert out2 == ["then turn"] and rest2 == ""


def test_split_sentences_arabic_question():
    out, _ = split_sentences("أضيفها؟ تمام.", force=True)
    assert out == ["أضيفها؟", "تمام."]


def test_speak_minutes_arabic_plurals():
    assert speak_minutes(1, "ar") == "دقيقة"
    assert speak_minutes(2, "ar") == "دقيقتين"
    assert speak_minutes(7, "ar") == "7 دقايق"
    assert speak_minutes(25, "ar") == "25 دقيقة"
    assert "ساعتين" in speak_minutes(125, "ar")


def test_speak_distance_no_decimals_spoken():
    for km in (0.3, 1.4, 2.0, 7.5, 12.3, 28.0):
        for lang in ("ar", "en"):
            s = speak_distance(km, lang)
            assert s is not None
            # the only digits allowed are integers (no "1.4"-style decimals)
            assert not re.search(r"\d+\.\d", s), f"{km} {lang} → {s}"
