"""
api/copilot_lang.py — CopilotV2 language core (pure, dependency-free, unit-tested).

THE contract (one sentence): exactly ONE resolve_language() decision is made per
turn, and that single value drives the prompt pin, the tool formatting, the
output validator, the TTS voice and every fallback string — nothing downstream
is allowed to re-guess.

Fixes the shipped v1 failure modes:
  1. Arabizi ("ana 3ayez asra3 tare2") — pure-Latin script, so the old
     dominant-script rule replied in ENGLISH to an Arabic sentence.
  2. Borrowed Latin tokens («خدني عالring road») — "ring road" outweighed the
     Arabic verb half the time; the reply language flip-flopped mid-trip.
  3. No stickiness — a garbled/empty STT turn fell back to app_lang and the
     conversation suddenly switched language ("multi-turn drift").
  4. No output validation — when the model ignored the language instruction the
     mismatched reply went straight to TTS. reply_lang_ok() is the deterministic
     gate the stream now runs BEFORE any text reaches the client.

Everything here is pure Python (no I/O, no framework imports) so the 120+ case
language harness runs it verbatim: tests/test_copilot_language.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── Script counting ───────────────────────────────────────────────────────────
# Arabic letters across the blocks STT/typed input actually produces.
_AR_RANGES = (
    ("؀", "ۿ"),   # Arabic
    ("ݐ", "ݿ"),   # Arabic Supplement
    ("ࢠ", "ࣿ"),   # Arabic Extended-A
    ("ﭐ", "﷿"),   # Presentation Forms-A
    ("ﹰ", "﻿"),   # Presentation Forms-B
)
# Arabic-Indic digits and Arabic punctuation are NOT letters — a lone «؟» must
# never make a turn "Arabic".
_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
_AR_PUNCT = "؟،؛٪٫٬٭ـ«»"


def _is_ar_letter(c: str) -> bool:
    if c in _AR_DIGITS or c in _AR_PUNCT:
        return False
    return any(a <= c <= b for a, b in _AR_RANGES)


def _is_en_letter(c: str) -> bool:
    return c.isascii() and c.isalpha()


def script_counts(text: str) -> Tuple[int, int]:
    """(arabic_letters, latin_letters) in the text."""
    ar = sum(1 for c in text if _is_ar_letter(c))
    en = sum(1 for c in text if _is_en_letter(c))
    return ar, en


# ── Borrowed tokens (stripped before the script ratio) ────────────────────────
# Latin words Egyptians drop inside Arabic sentences WITHOUT switching language:
# road names, chains, and everyday loanwords. «خدني عالring road» is an ARABIC
# sentence — "ring road" is vocabulary, not a language switch. Stripping these
# from the Latin count makes the ratio measure the sentence's real language.
_BORROWED_LATIN = [
    # roads / places
    "ring road", "ring", "mehwar", "autostrad", "autostrad road", "corniche",
    "downtown", "tagamo", "tagamoa", "zayed", "october", "maadi", "nasr city",
    "heliopolis", "madinaty", "rehab", "shorouk", "obour",
    # fuel / food / retail chains (mirror of copilot.py _BRANDS Latin aliases)
    "master", "chillout", "chill out", "on the run", "circle k", "wataniya",
    "watanya", "misr petroleum", "totalenergies", "total", "mobil",
    "emarat misr", "cilantro", "costa", "starbucks", "dunkin", "beano",
    "beanos", "mcdonald", "mcdonalds", "kfc", "momen", "cook door",
    "buffalo burger", "el ezaby", "ezaby", "seif",
    # interjection loans only — GENERIC English nouns ("gas station", "mall",
    # "coffee") deliberately stay OUT: stripping those would bias an English
    # sentence toward Arabic. Proper nouns + pure interjections only.
    "ok", "okay",
]
# Arabic-script tokens an ENGLISH sentence may legitimately carry (place/brand
# names Google or the user injects) — symmetric strip for the en side.
_BORROWED_ARABIC = [
    "ماستر", "تشيل اوت", "شيل اوت", "اون ذا رن", "سيركل ك", "الوطنيه", "وطنيه",
    "مصر للبترول", "توتال", "موبيل", "امارات مصر", "سيلانترو", "كوستا",
    "ستاربكس", "دانكن", "بينوس", "ماكدونالدز", "كنتاكي", "مؤمن", "كوك دور",
    "بافلو برجر", "العزبي", "عزبي", "صيف", "الدائري", "المحور", "الاوتوستراد",
    "التجمع الخامس", "التجمع", "الشيخ زايد", "مدينتي", "الرحاب", "الشروق",
    "العبور", "مدينه نصر", "مصر الجديده", "المعادي", "وسط البلد",
    "العاصمه الاداريه", "اكتوبر", "الهرم", "المهندسين", "الزمالك",
]


def _norm_for_match(s: str) -> str:
    s = s.lower()
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي"), ("ـ", "")):
        s = s.replace(a, b)
    return s


def _strip_tokens(text: str, tokens: List[str]) -> str:
    t = _norm_for_match(text)
    # longest-first so "ring road" is removed before "ring"
    for tok in sorted(tokens, key=len, reverse=True):
        t = t.replace(_norm_for_match(tok), " ")
    return t


# ── Arabizi detection (Latin-script Arabic) ───────────────────────────────────
# Two signals:
#   a) digits used as LETTERS inside a word (3ayez=عايز, tare2=طريق, za7ma) —
#      the strongest marker; shaped so English ordinals/units ("2nd", "8am",
#      "5km", "mp3") can NEVER trip it: a digit only counts sandwiched between
#      letters, or leading/trailing a run of ≥3 letters.
#   b) unmistakably-Egyptian romanized tokens.
_DIGIT_MID  = re.compile(r"[a-z][23579][a-z]")   # ta7t, za7ma
_DIGIT_LEAD = re.compile(r"^[23579][a-z]{3,}")   # 3ayez, 7elwa  (not "2nd")
_DIGIT_TAIL = re.compile(r"[a-z]{3,}[23579]$")   # tare2, sawa2  (not "mp3")

# Tokens individually decisive: sentence-forming Egyptian Arabic no English
# speaker produces. (Loanwords English speakers DO borrow — yalla, khalas,
# tamam — live in the COMMON set and need a second witness.)
_ARABIZI_STRONG = {
    "3ayez", "3ayz", "3awez", "3ayza", "3awza", "3aiz", "za7ma", "zahma",
    "ezayak", "ezzayak", "fein", "feen", "mafeesh", "mafish", "ma3lesh",
    "keda", "kda", "delwa2ty", "delwaty", "asra3", "tare2", "taree2", "tari2",
    "benzeena", "banzeena", "banzeen", "wareeni", "warini", "wadini",
    "haro7", "aro7", "nro7", "erga3", "hatly", "3ala", "3la", "5od",
}
# Common romanized-Egyptian words: two or more together mean Arabizi
# (individually they collide with English/loan words: "ana", "law", "bas"...).
_ARABIZI_COMMON = {
    "ana", "enta", "enty", "howa", "heya", "ehna", "homa", "mesh", "mush",
    "msh", "eh", "leh", "lama", "law", "lw", "bas", "tab", "tayeb", "tyb",
    "mashy", "mashi", "tamam", "momken", "mumkin", "fen", "wen", "yalla",
    "yala", "khalas", "5alas", "elly", "elli", "aiwa", "aywa", "la2", "la2a",
    "khod", "sheel", "shil", "hat", "raga3", "shwaya", "shwya", "roo7",
    "shar3", "midan", "kobri", "kobry", "ganb", "gamb", "ba3d", "abl",
    "wara", "odam", "yemin", "yameen", "shmal", "shemal", "ya3ni",
    "basha", "kamel", "kamera", "balak",
}
_WORD_RE = re.compile(r"[a-z0-9']+")


def _digit_letter_word(w: str) -> bool:
    return bool(_DIGIT_MID.search(w) or _DIGIT_LEAD.search(w)
                or _DIGIT_TAIL.search(w))


def is_arabizi(text: str) -> bool:
    """True when a Latin-script text is actually Arabic written in Latin
    letters. Only meaningful for texts with little/no Arabic script."""
    t = _norm_for_match(text)
    words = _WORD_RE.findall(t)
    if not words:
        return False
    digit_hits = sum(1 for w in words if _digit_letter_word(w))
    strong = sum(1 for w in words if w in _ARABIZI_STRONG)
    common = sum(1 for w in words if w in _ARABIZI_COMMON)
    if digit_hits >= 1 or strong >= 1:
        return True
    # Two common markers, or one common marker dominating a very short utterance
    # ("yalla bina", "ana gay"): still Arabizi.
    if common >= 2:
        return True
    if common == 1 and len(words) <= 2:
        return True
    return False


# ── Explicit language requests ────────────────────────────────────────────────
_EXPLICIT_AR = re.compile(
    r"(بالعربي|بالعربى|بالعربية|اتكلم عربي|كلمني عربي|عربي لو سمحت"
    r"|speak arabic|in arabic|talk arabic|answer in arabic|arabic please)")
_EXPLICIT_EN = re.compile(
    r"(بالانجليزي|بالإنجليزي|بالانجليزية|بالإنجليزية|اتكلم انجليزي|كلمني انجليزي"
    r"|speak english|in english|talk english|answer in english|english please)")


def explicit_language_request(text: str) -> Optional[str]:
    t = _norm_for_match(text)
    # check EN-request first: «بالانجليزي» contains Arabic script but asks for EN
    if _EXPLICIT_EN.search(t):
        return "en"
    if _EXPLICIT_AR.search(t):
        return "ar"
    return None


# ── THE resolver ──────────────────────────────────────────────────────────────
@dataclass
class ResolvedLang:
    lang: str        # "ar" | "en" — the single authoritative value for the turn
    source: str      # explicit | arabizi | script | sticky | fallback
    arabizi: bool    # input was Latin-script Arabic (model must be told)


def resolve_language(text: str,
                     prev_lang: Optional[str] = None,
                     app_lang: str = "en") -> ResolvedLang:
    """The ONE language decision for a turn.

    Order of authority:
      1. An explicit request ("بالعربي" / "speak English") wins outright.
      2. Arabizi detection (Latin-script Arabic) → Arabic.
      3. Script ratio AFTER stripping borrowed tokens; a clear majority (≥60%)
         wins; the 40–60% gray zone stays sticky on prev_lang.
      4. No letters at all → prev_lang → app_lang → "en".
    """
    text = (text or "").strip()
    prev = prev_lang if prev_lang in ("ar", "en") else None
    fallback = prev or (app_lang if app_lang in ("ar", "en") else "en")
    if not text:
        return ResolvedLang(fallback, "fallback", False)

    exp = explicit_language_request(text)
    if exp:
        return ResolvedLang(exp, "explicit", False)

    ar_raw, en_raw = script_counts(text)
    if en_raw > 0 and ar_raw == 0 and is_arabizi(text):
        return ResolvedLang("ar", "arabizi", True)

    # strip borrowed vocabulary so loanwords can't flip the sentence language
    ar, _ = script_counts(_strip_tokens(text, _BORROWED_ARABIC)) if ar_raw else (0, 0)
    _, en = script_counts(_strip_tokens(text, _BORROWED_LATIN)) if en_raw else (0, 0)
    if ar == 0 and en == 0:
        # everything was borrowed tokens/digits — keep the conversation language
        if ar_raw or en_raw:
            # raw counts break the tie when no history exists
            raw_guess = "ar" if ar_raw >= en_raw else "en"
            return ResolvedLang(prev or raw_guess, "sticky" if prev else "script", False)
        return ResolvedLang(fallback, "fallback", False)
    if ar == 0:
        return ResolvedLang("en", "script", False)
    if en == 0:
        return ResolvedLang("ar", "script", False)
    ratio = ar / (ar + en)
    if ratio >= 0.60:
        return ResolvedLang("ar", "script", False)
    if ratio <= 0.40:
        return ResolvedLang("en", "script", False)
    # 40–60% gray zone. Two or more real Arabic words surviving the strip is
    # an Arabic sentence FRAME carrying English nouns («عايز اروح Mall of
    # Egypt») — that's a genuine Arabic turn even mid-English conversation.
    # Otherwise stay sticky; with no history, lean Arabic (the common case for
    # Egyptian mixing).
    ar_words = sum(1 for w in _strip_tokens(text, _BORROWED_ARABIC).split()
                   if sum(1 for c in w if _is_ar_letter(c)) >= 2)
    if ar_words >= 2:
        return ResolvedLang("ar", "script", False)
    if prev:
        return ResolvedLang(prev, "sticky", False)
    return ResolvedLang("ar", "script", False)


# ── Output validator ──────────────────────────────────────────────────────────
# Deterministic gate run on the backend BEFORE any generated text is emitted.
# A reply passes for lang=ar when, after dropping digits/punct and borrowed
# Latin tokens (brands, road names — legitimately Latin inside Arabic), Arabic
# letters are the majority — and symmetrically for en. Short name-like chunks
# ("Master.") are exempt: they carry no language.

def reply_lang_ratio(text: str, lang: str) -> float:
    """Fraction of letters in the TARGET language's script, borrowed tokens and
    digits excluded. 1.0 for text with no letters at all (nothing to violate)."""
    if lang == "ar":
        stripped = _strip_tokens(text, _BORROWED_LATIN)
    else:
        stripped = _strip_tokens(text, _BORROWED_ARABIC)
    ar, en = script_counts(stripped)
    total = ar + en
    if total == 0:
        return 1.0
    return (ar / total) if lang == "ar" else (en / total)


def _name_like(text: str) -> bool:
    """A chunk that is mostly a proper name / brand — exempt from validation.
    ≤4 words and no sentence-forming function words of the WRONG language."""
    words = re.findall(r"[^\s]+", text.strip())
    if len(words) > 4:
        return False
    t = " " + _norm_for_match(text) + " "
    for w in (" the ", " is ", " are ", " in ", " on ", " at ", " to ",
              " and ", " it ", " you ", " your ", " there "):
        if w in t:
            return False
    return True


def reply_lang_ok(text: str, lang: str, threshold: float = 0.5) -> bool:
    """True when the reply text matches the resolved language."""
    if not text or not text.strip():
        return True
    if reply_lang_ratio(text, lang) >= threshold:
        return True
    return _name_like(text)


# ── Sentence splitting (mirrors the clients' decimal-safe chunker) ────────────
_BOUNDARY = ".!?؟…\n"


def split_sentences(buf: str, force: bool) -> Tuple[List[str], str]:
    """Extract complete sentences from a streaming buffer. A boundary char only
    counts when followed by whitespace/end so decimals ("2.5 km") stay whole.
    Returns (complete_sentences, remaining_buffer); force flushes the tail."""
    out: List[str] = []
    while True:
        cut = -1
        for i, c in enumerate(buf):
            if c in _BOUNDARY:
                if i == len(buf) - 1:
                    if force:
                        cut = i
                    break
                if buf[i + 1].isspace():
                    cut = i
                    break
        if cut < 0:
            break
        sentence = buf[: cut + 1].strip()
        buf = buf[cut + 1:]
        if sentence:
            out.append(sentence)
    if force:
        rest = buf.strip()
        buf = ""
        if rest:
            out.append(rest)
    return out, buf


# ── Speech-natural numbers (shared by fast-path templates) ────────────────────
def speak_minutes(minutes: int, lang: str) -> str:
    """Verbalize a duration the way a person says it — Arabic dual/plural rules
    respected, hours split out past 90 min."""
    m = max(0, int(round(minutes)))
    if lang == "ar":
        if m >= 90:
            h, rem = divmod(m, 60)
            hh = "ساعة" if h == 1 else ("ساعتين" if h == 2 else f"{h} ساعات")
            if rem >= 25 and rem <= 35:
                return f"{hh} ونص"
            if rem >= 10:
                return f"{hh} و{speak_minutes(rem, 'ar')}"
            return hh
        if m == 0:
            return "أقل من دقيقة"
        if m == 1:
            return "دقيقة"
        if m == 2:
            return "دقيقتين"
        if m <= 10:
            return f"{m} دقايق"
        return f"{m} دقيقة"
    if m >= 90:
        h, rem = divmod(m, 60)
        hh = "an hour" if h == 1 else f"{h} hours"
        if 25 <= rem <= 35:
            return f"{hh} and a half"
        if rem >= 10:
            return f"{hh} and {rem} minutes"
        return hh
    if m == 0:
        return "under a minute"
    if m == 1:
        return "a minute"
    return f"{m} minutes"


def speak_distance(km: Optional[float], lang: str) -> Optional[str]:
    """Round a distance for the EAR (no decimals TTS mangles)."""
    if km is None:
        return None
    if km < 0.975:
        m = max(50, int(round(km * 1000 / 50.0)) * 50)
        return f"{m} متر" if lang == "ar" else f"{m} meters"
    if km < 9.75:
        halves = round(km * 2) / 2
        if lang == "ar":
            whole = int(halves)
            if halves == whole:
                return "كيلومتر" if whole == 1 else (
                    "كيلومترين" if whole == 2 else f"{whole} كيلومترات"
                    if whole <= 10 else f"{whole} كيلومتر")
            if whole == 0:
                return "نص كيلو"
            if whole == 1:
                return "كيلو ونص"
            return f"{whole} كيلو ونص"
        whole = int(halves)
        if halves == whole:
            return "a kilometer" if whole == 1 else f"{whole} kilometers"
        if whole == 0:
            return "half a kilometer"
        if whole == 1:
            return "a kilometer and a half"
        return f"{whole} and a half kilometers"
    n = int(round(km))
    return f"{n} كيلومتر" if lang == "ar" else f"{n} kilometers"
