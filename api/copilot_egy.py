# -*- coding: utf-8 -*-
"""
api/copilot_egy.py — Egyptian verbalization of Arabic assistant text (pure).

WHY THIS EXISTS: a large part of "the Arabic voice sounds formal/pan-Arab" is
not the voice — it is the TEXT. Every Arabic TTS engine reads the digit string
"80" as the Modern-Standard numeral «ثمانون», "17" as «سبعة عشر» and "5:57" as
«خمسة وسبعة وخمسون دقيقة». A Cairene says «تمانين», «سبعتاشر» and «ستة إلا
تلاتة». Spelling the numbers out in Egyptian colloquial BEFORE synthesis makes
any engine speak them the Egyptian way — deterministically, on every path,
regardless of what the model wrote. This runs inside the single output gate
(api/copilot_v2.py Emitter) for every Arabic line: model output, fast-path
templates, catalog strings alike.

Rules (Cairene colloquial, as spoken):
  • 0–19 have their own words (تلاتة/تلتاشر, تمانية/تمنتاشر …)
  • tens: عشرين تلاتين أربعين خمسين ستين سبعين تمانين تسعين
  • compounds are unit-then-ten: 25 → «خمسة وعشرين»
  • hundreds: مية / ميتين / تلتمية … ; thousands: ألف / ألفين / تلاتة آلاف
  • 3–10 directly before a NOUN take the short form: «تلات رادارات»,
    «خمس دقايق», «عشر كيلو» (the "counting" form); standalone → «تلاتة»
  • 1 and 2 before a noun are dropped/dualized by the model already
    («رادار واحد», «رادارين»); a literal "2 كيلو" becomes «كيلوين»
  • clock times «5:57» → «ستة إلا تلاتة»; «5:15» → «خمسة وربع»;
    «5:30» → «خمسة ونص»; «5:45» → «ستة إلا ربع»; «5:20» → «خمسة وتلت»;
    «5:40» → «ستة إلا تلت»; «5:05» → «خمسة وخمسة»; 24h → 12h
  • decimals «1.5» → «واحد ونص»; «2.5 كيلو» → «كيلوين ونص»
  • percent «30%» → «تلاتين في المية»
Latin/English text and digits inside brand names (19011, Route 75M) are left
alone: only digits followed by Arabic context or standing alone are touched.
"""

from __future__ import annotations

import re

_ONES = ["صفر", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة",
         "تمانية", "تسعة", "عشرة", "حداشر", "اتناشر", "تلتاشر", "أربعتاشر",
         "خمستاشر", "ستاشر", "سبعتاشر", "تمنتاشر", "تسعتاشر"]
# counting form used directly before a noun (3–10)
_ONES_COUNT = {3: "تلات", 4: "أربع", 5: "خمس", 6: "ست", 7: "سبع", 8: "تمن",
               9: "تسع", 10: "عشر"}
_TENS = {20: "عشرين", 30: "تلاتين", 40: "أربعين", 50: "خمسين", 60: "ستين",
         70: "سبعين", 80: "تمانين", 90: "تسعين"}
_HUNDREDS = {100: "مية", 200: "ميتين", 300: "تلتمية", 400: "ربعمية",
             500: "خمسمية", 600: "ستمية", 700: "سبعمية", 800: "تمنمية",
             900: "تسعمية"}


def number_words(n: int, before_noun: bool = False) -> str:
    """Egyptian colloquial words for a non-negative integer (≤ 999,999)."""
    if n < 0:
        return "ناقص " + number_words(-n, before_noun)
    if n < 20:
        if before_noun and n in _ONES_COUNT:
            return _ONES_COUNT[n]
        return _ONES[n]
    if n < 100:
        t, u = (n // 10) * 10, n % 10
        return _TENS[t] if u == 0 else f"{_ONES[u]} و{_TENS[t]}"
    if n < 1000:
        h, r = (n // 100) * 100, n % 100
        return _HUNDREDS[h] if r == 0 else f"{_HUNDREDS[h]} و{number_words(r)}"
    if n < 1_000_000:
        k, r = n // 1000, n % 1000
        if k == 1:
            head = "ألف"
        elif k == 2:
            head = "ألفين"
        elif k <= 10:
            head = f"{_ONES_COUNT.get(k, _ONES[k])} آلاف"
        else:
            head = f"{number_words(k)} ألف"
        return head if r == 0 else f"{head} و{number_words(r)}"
    return str(n)


def clock_words(h: int, m: int) -> str:
    """«5:57» → «ستة إلا تلاتة» — the way Egyptians tell the time."""
    h12 = h % 12 or 12
    nxt = (h12 % 12) + 1
    hw = _ONES[h12] if h12 != 2 else "اتنين"
    if m == 0:
        return hw
    if m == 15:
        return f"{hw} وربع"
    if m == 20:
        return f"{hw} وتلت"
    if m == 30:
        return f"{hw} ونص"
    if m == 40:
        return f"{_ONES[nxt]} إلا تلت"
    if m == 45:
        return f"{_ONES[nxt]} إلا ربع"
    if m < 30:
        return f"{hw} و{number_words(m)}"
    return f"{_ONES[nxt]} إلا {number_words(60 - m)}"


_AR_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_CLOCK = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_PERCENT = re.compile(r"(\d+)\s*[%٪]")
_DECIMAL = re.compile(r"(?<![\d.])(\d+)[.,٫](\d)(?![\d.])")
# a bare integer followed (after optional space) by an Arabic word → counting form
_INT_BEFORE_NOUN = re.compile(r"(?<![\w.:])(\d{1,6})(?=\s*[؀-ۿ])")
_INT_ALONE = re.compile(r"(?<![\w.:%])(\d{1,6})(?![\d:%]|\.\d|\s*[A-Za-z])")
# numeric names that must stay numeric (brands, phone numbers)
_KEEP_NUMERIC = re.compile(r"\b19011\b|\b\d{7,}\b")
_LATIN_CONTEXT = re.compile(r"[A-Za-z]\s*\d|\d\s*[A-Za-z]")


def _protect_latin_numbers(text: str):
    """Digits glued to Latin letters (19011, Route 75M, B2) are names — keep."""
    spans = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9]*\d[A-Za-z0-9]*|\d+[A-Za-z]+[A-Za-z0-9]*", text):
        spans.append(m.span())
    for m in _KEEP_NUMERIC.finditer(text):
        spans.append(m.span())
    return spans


def _in_spans(pos: int, spans) -> bool:
    return any(a <= pos < b for a, b in spans)


def egyptianize(text: str) -> str:
    """Rewrite digits/times/decimals in an ARABIC line as Egyptian words."""
    if not text:
        return text
    t = text.translate(_AR_DIGIT_MAP)
    protected = _protect_latin_numbers(t)

    def sub_clock(m):
        if _in_spans(m.start(), protected):
            return m.group(0)
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            return m.group(0)
        return clock_words(h, mi)
    t = _CLOCK.sub(sub_clock, t)

    def sub_percent(m):
        return f"{number_words(int(m.group(1)))} في المية"
    t = _PERCENT.sub(sub_percent, t)

    def sub_decimal(m):
        if _in_spans(m.start(), protected):
            return m.group(0)
        whole, frac = int(m.group(1)), int(m.group(2))
        if frac == 5:
            if whole == 0:
                return "نص"
            if whole == 1:
                return "واحد ونص"
            if whole == 2:
                return "اتنين ونص"
            return f"{number_words(whole)} ونص"
        if frac == 0:
            return number_words(whole)
        return f"{number_words(whole)} فاصل {number_words(frac)}"
    t = _DECIMAL.sub(sub_decimal, t)

    # «2 كيلو» → «كيلوين»; «1 كيلو» → «كيلو»  (dual/singular by noun)
    t = re.sub(r"(?<![\w.:])2\s+كيلو(?:متر)?\b", "كيلوين", t)
    t = re.sub(r"(?<![\w.:])1\s+كيلو(?:متر)?\b", "كيلو", t)
    t = re.sub(r"(?<![\w.:])2\s+دقيق[ةه]\b", "دقيقتين", t)
    t = re.sub(r"(?<![\w.:])1\s+دقيق[ةه]\b", "دقيقة", t)
    t = re.sub(r"(?<![\w.:])2\s+ساع[ةه]\b", "ساعتين", t)
    t = re.sub(r"(?<![\w.:])1\s+ساع[ةه]\b", "ساعة", t)

    def sub_before_noun(m):
        if _in_spans(m.start(), protected):
            return m.group(0)
        return number_words(int(m.group(1)), before_noun=True)
    t = _INT_BEFORE_NOUN.sub(sub_before_noun, t)

    def sub_alone(m):
        if _in_spans(m.start(), protected):
            return m.group(0)
        return number_words(int(m.group(1)))
    t = _INT_ALONE.sub(sub_alone, t)
    return t


# ── MSA → Egyptian wording of things the MODEL keeps writing formally ────────
# A small, deterministic dictionary of the formal words that slip into replies
# (and into older catalog strings). WHOLE-WORD only (Arabic-letter boundaries),
# longest key first. Not a translator — it catches the handful of fossils
# that make a line sound like a news anchor, and the two direction words
# Egyptians say differently («يسار» is «شمال» in Cairo).
_MSA_TO_EGY = {
    "حركة المرور": "الزحمة", "لا يوجد": "مفيش", "لا توجد": "مفيش",
    "الازدحام": "الزحمة", "ازدحام": "زحمة", "مزدحم": "زحمة", "مزدحمة": "زحمة",
    "الوجهة": "المكان اللي رايحه", "وجهتك": "المكان اللي رايحه",
    "ماذا": "ايه", "لماذا": "ليه", "متى": "امتى", "أين": "فين", "كيف": "ازاي",
    "الآن": "دلوقتي", "حاليًا": "دلوقتي", "حالياً": "دلوقتي", "حاليا": "دلوقتي",
    "جدًا": "أوي", "جداً": "أوي", "قليلًا": "شوية", "قليلاً": "شوية",
    "أيضًا": "كمان", "أيضاً": "كمان", "أيضا": "كمان", "كذلك": "كمان",
    "ولكن": "بس", "لكن": "بس", "سيارة": "عربية", "السيارة": "العربية",
    "سيارتك": "عربيتك", "يوجد": "فيه", "توجد": "فيه", "هناك": "فيه",
    "ليس": "مش", "ليست": "مش", "نعم": "ايوه", "أرغب": "عايز", "تريد": "عايز",
    "أريد": "عايز", "دقائق": "دقايق", "الدقائق": "الدقايق", "ثوان": "ثواني",
    "ثوانٍ": "ثواني", "كيلومترات": "كيلو", "كيلومتر": "كيلو", "أمتار": "متر",
    "اليسار": "الشمال", "يسار": "شمال", "يسارًا": "شمال", "يساراً": "شمال",
    "يمينًا": "يمين", "يميناً": "يمين", "بالتأكيد": "أكيد", "بالطبع": "أكيد",
    "رائع": "حلو", "ممتاز": "تمام", "للأسف": "معلش", "عذرًا": "معلش",
    "عذراً": "معلش", "آسف": "معلش", "أعتذر": "معلش", "الرجاء": "لو سمحت",
    "من فضلك": "لو سمحت", "رجاءً": "لو سمحت",
}
_FUTURE_RE = re.compile(r"(?<![؀-ۿ])سوف\s+(?=[يتنأا][؀-ۿ])")
_MSA_RE = re.compile(
    "(?<![؀-ۿ])(" + "|".join(re.escape(k) for k in
                             sorted(_MSA_TO_EGY, key=len, reverse=True)) + ")(?![؀-ۿ])")


def masri(text: str) -> str:
    """Egyptianize the wording (formal fossils) AND the numbers of an Arabic
    line. Idempotent."""
    if not text:
        return text
    t = _FUTURE_RE.sub("ه", text)                 # «سوف يكون» → «هيكون»
    t = _MSA_RE.sub(lambda m: _MSA_TO_EGY[m.group(1)], t)
    return egyptianize(t)
