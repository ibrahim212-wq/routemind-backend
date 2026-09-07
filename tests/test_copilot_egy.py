# -*- coding: utf-8 -*-
"""Egyptian verbalization — the deterministic half of "the voice must sound
Egyptian": numbers, clock times, decimals, direction words, formal fossils."""
import pytest
from api.copilot_egy import masri, egyptianize, number_words, clock_words


@pytest.mark.parametrize("n,words", [
    (0, "صفر"), (1, "واحد"), (2, "اتنين"), (3, "تلاتة"), (8, "تمانية"),
    (11, "حداشر"), (13, "تلتاشر"), (17, "سبعتاشر"), (18, "تمنتاشر"),
    (20, "عشرين"), (25, "خمسة وعشرين"), (80, "تمانين"), (99, "تسعة وتسعين"),
    (100, "مية"), (112, "مية واتناشر"), (200, "ميتين"), (300, "تلتمية"),
    (450, "ربعمية وخمسين"), (1000, "ألف"), (1500, "ألف وخمسمية"),
    (2000, "ألفين"), (3000, "تلات آلاف"),
])
def test_number_words(n, words):
    assert number_words(n) == words


def test_counting_form_before_noun():
    assert number_words(3, before_noun=True) == "تلات"
    assert number_words(8, before_noun=True) == "تمن"
    assert number_words(10, before_noun=True) == "عشر"
    assert number_words(11, before_noun=True) == "حداشر"    # no short form


@pytest.mark.parametrize("h,m,words", [
    (5, 0, "خمسة"), (5, 15, "خمسة وربع"), (5, 20, "خمسة وتلت"),
    (5, 30, "خمسة ونص"), (5, 40, "ستة إلا تلت"), (5, 45, "ستة إلا ربع"),
    (5, 5, "خمسة وخمسة"), (17, 57, "ستة إلا تلاتة"), (18, 30, "ستة ونص"),
    (12, 0, "اتناشر"), (0, 10, "اتناشر وعشرة"), (23, 50, "اتناشر إلا عشرة"),
])
def test_clock_words(h, m, words):
    assert clock_words(h, m) == words


def test_line_numbers_become_egyptian_words():
    out = masri("قدامك 3 رادارات — أقربهم بعد 1.5 كيلو، والسرعة عنده 80.")
    assert "تلات رادارات" in out and "واحد ونص كيلو" in out and "تمانين" in out
    assert not any(ch.isdigit() for ch in out)


def test_clock_in_line():
    assert "ستة إلا تلاتة" in masri("هتوصل حوالي 17:57.")
    assert "خمسة وربع" in masri("الساعة 5:15")


def test_dual_and_singular_units():
    assert masri("بعد 2 كيلو") == "بعد كيلوين"
    assert masri("بعد 1 كيلو") == "بعد كيلو"
    assert masri("فاضل 2 دقيقة") == "فاضل دقيقتين"


def test_numeric_brands_and_phone_numbers_untouched():
    out = masri("أقرب صيدلية 19011 على بعد 400 متر، اتصل على 01001234567.")
    assert "19011" in out and "01001234567" in out and "ربعمية متر" in out


def test_latin_names_untouched():
    out = masri("Master على بعد 2 كيلو، Route 75M بعد 1 كيلو.")
    assert "Master" in out and "Route 75M" in out and "كيلوين" in out


def test_formal_fossils_and_directions():
    out = masri("الطريق مزدحم الآن ولكن سوف يكون أفضل بعد 25 دقيقة.")
    assert "زحمة" in out and "دلوقتي" in out and "بس" in out and "هيكون" in out
    assert "خمسة وعشرين دقيقة" in out
    assert masri("خد يسارًا بعد 300 متر ثم يمينًا.") == "خد شمال بعد تلتمية متر ثم يمين."


def test_whole_word_only():
    # «هل» inside «أهلاً» must not be touched; «كيف» inside a name neither
    assert masri("أهلاً بيك") == "أهلاً بيك"


def test_idempotent():
    line = "قدامك 3 رادارات بعد 1.5 كيلو، السرعة 80، هتوصل 17:57."
    once = masri(line)
    assert masri(once) == once


def test_percent():
    assert "تلاتين في المية" in masri("أتقل بنسبة 30% من العادي")


def test_english_line_is_never_passed_here_but_survives():
    # the gate only calls masri for lang=ar; if it ever received English it
    # must not corrupt it
    assert masri("Heavy traffic in about 2 kilometers") == "Heavy traffic in about 2 kilometers"
