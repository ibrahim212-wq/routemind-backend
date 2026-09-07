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
_DIGIT_LEAD2 = re.compile(r"^[237][a-z]{2}$")    # 2ol (قول), 3an (عن), 7ad (حد)
_DIGIT_TAIL2 = re.compile(r"^[a-z]{2}[27]$")     # la2 (لأ), ba7
_ENGLISH_ORDINALS = {"2nd", "3rd"}

# Tokens individually decisive: sentence-forming Egyptian Arabic no English
# speaker produces. (Loanwords English speakers DO borrow — yalla, khalas,
# tamam — live in the COMMON set and need a second witness.)
_ARABIZI_STRONG = {
    "3ayez", "3ayz", "3awez", "3ayza", "3awza", "3aiz", "za7ma", "zahma",
    "ezayak", "ezzayak", "fein", "feen", "mafeesh", "mafish", "ma3lesh",
    "keda", "kda", "delwa2ty", "delwaty", "asra3", "tare2", "taree2", "tari2",
    "benzeena", "banzeena", "banzeen", "wareeni", "warini", "wadini",
    "haro7", "aro7", "nro7", "erga3", "hatly", "3ala", "3la", "5od",
    "emta", "emtaa", "hanewsal", "hanwsal", "newsal", "fadel", "fadl",
    "fadelly", "gheir", "ghair", "rosoom", "rosom", "asdy", "asdi", "2asdy",
    "mohandeseen", "mohandesen", "laffa", "lafa", "balash", "3adia", "3ady",
    "3amel", "3amla", "3aml", "arawa7", "rakna", "raken", "2odam", "2odamy",
    "oddam", "shwaya", "so2al", "3edt", "tolo2", "toro2", "saree3a", "saree3",
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
    "basha", "kamel", "kamera", "balak", "el", "kam", "tany", "tani",
    "eh", "dah", "dih", "di", "da", "beit", "bait", "gaya", "gay",
    "tare2", "sot", "2ol", "ol", "3alli", "warini",
}
_WORD_RE = re.compile(r"[a-z0-9']+")


def _digit_letter_word(w: str) -> bool:
    if w in _ENGLISH_ORDINALS:
        return False
    return bool(_DIGIT_MID.search(w) or _DIGIT_LEAD.search(w)
                or _DIGIT_TAIL.search(w) or _DIGIT_LEAD2.search(w)
                or _DIGIT_TAIL2.search(w))


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


# ── Evidence lexicons (v3: a language SWITCH needs recognizable words) ─────────
# A compact English vocabulary: function words + everyday speech + everything a
# driver says to a navigation assistant. Membership = "this Latin token is a
# real English word". Unknown Latin tokens (names, STT junk) are NOT evidence
# of English; they can't flip an Arabic conversation on their own.
_EN_WORDS = set("""
a about above accident add address after again ahead all almost alone along
already also alternate alternative always am an and another answer any anymore
anything are area around arrive arrived arriving as ask at ate atm avoid away
back bad be because been before behind best better big bit bridge bring bus by
cafe call camera cameras can cancel car card care cash change charge charging
check city clear close closed closest coffee cold come coming confirm cost could
cross current day delay destination detour did different direction directions
directly distance do does doing done dont don't down drive driver driving drop
during each early eat eight eighteen eighty either else end enough eta even
evening ever every exit expect fast faster fastest few fifteen fifty find fine
first five flyover follow food for forty four free from fuel full further gas
get getting give go going gone good got great guess had half hand happen has
have having he hear heavy help her here hey hi highway him his hit hold home
hospital hotel hour hours how hungry i if in inside instead into is it its jam
just keep kilometer kilometers kilometre km know last late later lane left less
let light like limit little long look looking lot loud louder low lower make
many map maps me mean mechanic meter meters mile miles minute minutes mode more
morning mosque most motorway move much mute my name navigate navigation near
nearby nearest need never new next nice night nine ninety no normal not nothing
now number of off office ok okay old on once one only open option options or
other our out over park parking past pay people petrol pharmacy phone pick place
places play please point police previous quiet quieter quick quicker quickest
radar rain ready really recalculate remember remind reminder remove repeat
report reroute rest restaurant right ring road roads route routes run same save
saved say school search second see seem send set seven seventeen seventy share
she shop shopping should show side since sixty six sixteen skip slow slower
small so some someone something soon sorry sound speak speed speeding start
station stay still stop stops store street sure switch take talk tell ten than
thank thanks that the their them then there these they thing think third
thirteen thirty this those three through time to today toll tolls too took top
total traffic trip try turn twelve twenty two under unmute until up us use
usual usually very voice wait want was watch water way we weather week well
went were what when where which while who why will with without work worse
would wrong yeah year yes yesterday yet you your zero
""".split())

# Interjections/hesitations: neutral — neither evidence of English nor junk.
_NEUTRAL_LATIN = {"hmm", "hm", "mm", "mmm", "uh", "um", "umm", "ah", "eh", "oh",
                  "er", "erm", "huh", "aha", "hmmm", "ehh", "ahh"}

# English rendered in ARABIC SCRIPT — what the ar-EG recognizer emits when the
# driver actually spoke English into it ("هاو لونج ليفت"). Function words and
# core assistant verbs only; borrowed loanwords Egyptians really use inside
# Arabic («الترافيك», «كاميرا», «روت») are deliberately NOT here.
_TRANSLIT_EN = {
    "هاو", "وات", "وير", "وين", "واي", "هوين", "ذا", "ذي", "ذيس", "ذات",
    "إز", "از", "إيز", "آر", "دو", "دوز", "دونت", "بليز",
    "ثانكس", "ثانك", "يو", "مي", "ماي", "تو", "فور", "فروم", "أند", "اند",
    "بت", "بات", "نوت", "يس", "نو", "تيك", "شو", "تيل", "جيت", "جو", "جوينج",
    "تيرن", "نيكست", "لفت", "رايت", "ستوب", "كانسل", "ميوت", "أنميوت",
    "لاودر", "ريبيت", "سويتش", "أفويد", "اسكيب", "ريمايند", "شير", "كول",
    "هوم", "وورك", "ورك", "لونج", "ليفت", "فار", "مني", "ماني",
    "سبيد", "ليميت", "فاست", "فاستر", "فاستست", "روود", "ستريت", "هايواي",
    "تول", "تولز", "إكسيت", "اكسيت", "بريدج", "ستيشن", "بيتزا", "أوردر",
    "بوك", "فلايت", "تايم", "أرايف", "ارايف", "أرايفنج", "مينتس", "مينت",
}
_AR_WORD_RE = re.compile(r"[؀-ۿ]+")
# Two-letter Arabic tokens that ARE real words a driver says («لا» = no); any
# other two-letter fragment («ال», «ممم» after junk-filtering) is not evidence.
_AR_SHORT_WORDS = {"لا", "لأ", "اه", "آه", "ده", "دي", "في", "من", "مش", "لو",
                   "طب", "يا", "ما", "او", "أو", "هو", "هي", "كل", "عن", "بس"}


def _latin_tokens(text: str):
    return re.findall(r"[a-z][a-z']*", _norm_for_match(text))


def _is_junk_token(tok: str) -> bool:
    """A token with no plausible word shape: one distinct character repeated
    ('hhh', 'ااا', 'ةةة'), or a Latin run with no vowel at all ('asdkjh')."""
    if len(set(tok)) == 1 and len(tok) >= 2:
        return True
    if tok.isascii() and tok.isalpha() and len(tok) >= 3 \
            and not any(c in "aeiouy" for c in tok):
        return True
    return False


def english_evidence(text: str):
    """(real_english_words, latin_tokens_considered) after dropping borrowed
    proper nouns, neutral interjections and junk."""
    stripped = _strip_tokens(text, _BORROWED_LATIN)
    toks = [t for t in _latin_tokens(stripped) if not _is_junk_token(t)]
    # neutral hesitations ("uh", "hmm") stay in the DENOMINATOR (they dilute a
    # claim) but never count as real words: "mmm uh the uh" is 1 of 3.
    real = sum(1 for t in toks if t.strip("'") in _EN_WORDS)
    return real, len(toks)


def arabic_evidence(text: str):
    """(real_arabic_words, arabic_tokens_considered, transliterated_english)
    — an Arabic token counts as a real word when it has ≥2 distinct letters
    and isn't junk; tokens from the transliterated-English lexicon are
    counted separately (they are evidence of ENGLISH speech)."""
    stripped = _strip_tokens(text, _BORROWED_ARABIC)
    toks = _AR_WORD_RE.findall(stripped)
    translit = 0
    real = 0
    for t in toks:
        letters = [c for c in t if _is_ar_letter(c)]
        if len(letters) < 2 or _is_junk_token("".join(letters)):
            continue
        # transliterated-English lexicon first: its members include 2-letter
        # renderings («شو», «ذا», «تو») that the short-word filter below
        # would otherwise discard.
        core = t[2:] if t.startswith("ال") and len(t) > 4 else t
        if t in _TRANSLIT_EN or core in _TRANSLIT_EN:
            translit += 1
            continue
        if len(letters) == 2 and _norm_for_match("".join(letters)) not in                 {_norm_for_match(w) for w in _AR_SHORT_WORDS}:
            continue
        real += 1
    return real, len(toks), translit


def is_transliterated_english(text: str) -> bool:
    """Arabic script that is really English (the ar-EG recognizer heard an
    English sentence): ≥2 lexicon hits and they dominate the Arabic tokens."""
    real, n, translit = arabic_evidence(text)
    return translit >= 2 and translit >= 0.6 * max(1, real + translit)


# Below this STT confidence, a language that differs from the conversation's
# is not trusted to switch it: the mic was most likely open in the wrong
# language (Android has one recognizer; its wrong-language output is real-
# looking words at low confidence). The client's own cross-language retry
# handles the very-low-confidence case before it ever reaches us.
STT_SWITCH_MIN_CONF = 0.55


# ── THE resolver ──────────────────────────────────────────────────────────────
@dataclass
class ResolvedLang:
    lang: str        # "ar" | "en" — the single authoritative value for the turn
    source: str      # explicit | arabizi | translit | evidence | sticky | fallback
    arabizi: bool    # input was Latin-script Arabic (model must be told)
    unreliable: bool = False   # low-confidence / garbled: model should confirm briefly


def resolve_language(text: str,
                     prev_lang: Optional[str] = None,
                     app_lang: str = "en",
                     stt_lang: Optional[str] = None,
                     stt_confidence: Optional[float] = None) -> ResolvedLang:
    """The ONE language decision for a turn.

    v3 rule: a language SWITCH needs EVIDENCE — recognizable words of the new
    language. Gibberish, junk, numbers, names and hesitations never switch;
    they stay with the conversation. Order of authority:
      1. An explicit request ("بالعربي" / "speak English") wins outright.
      2. Arabizi (Latin-script Arabic) → Arabic; transliterated English
         (Arabic-script English) → English.
      3. Real-word evidence per language, borrowed tokens stripped; a clear
         majority wins; the gray zone → Arabic sentence frame → sticky.
      4. No evidence → prev_lang → app_lang → "en".
      5. A switch away from prev_lang under low STT confidence is refused
         (the recognizer was probably open in the wrong language).
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
    if ar_raw > 0 and is_transliterated_english(text):
        return _guard_switch(ResolvedLang("en", "translit", False, True),
                             prev, stt_lang, stt_confidence)

    en_real, en_n = english_evidence(text)
    ar_real, ar_n, _ = arabic_evidence(text)
    # An English claim needs real words AND a real share of its own tokens
    # ("mmm uh the uh" = 1 real of 3 considered → not English evidence).
    en_ok = en_real >= 1 and (en_real >= 2 or en_n <= 2) and en_real / max(1, en_n) >= 0.4
    ar_ok = ar_real >= 1

    if not en_ok and not ar_ok:
        # numbers, junk, names, hesitations only → the conversation language
        low = stt_confidence is not None and stt_confidence < STT_SWITCH_MIN_CONF
        return ResolvedLang(fallback, "sticky" if prev else "fallback", False,
                            unreliable=bool((ar_raw + en_raw) > 0 or low))
    if en_ok and not ar_ok:
        return _guard_switch(ResolvedLang("en", "evidence", False),
                             prev, stt_lang, stt_confidence)
    if ar_ok and not en_ok:
        return _guard_switch(ResolvedLang("ar", "evidence", False),
                             prev, stt_lang, stt_confidence)

    # both languages carry real words → the sentence FRAME (more real words)
    # wins; «عايز الفastest route» is an Arabic sentence with an English noun.
    if ar_real > en_real:
        return _guard_switch(ResolvedLang("ar", "evidence", False), prev, stt_lang, stt_confidence)
    if en_real > ar_real:
        return _guard_switch(ResolvedLang("en", "evidence", False), prev, stt_lang, stt_confidence)
    # equal word counts → letter ratio, borrowed tokens stripped
    ar, _ = script_counts(_strip_tokens(text, _BORROWED_ARABIC))
    _, en = script_counts(_strip_tokens(text, _BORROWED_LATIN))
    ratio = ar / max(1, ar + en)
    if ratio >= 0.60:
        return _guard_switch(ResolvedLang("ar", "evidence", False), prev, stt_lang, stt_confidence)
    if ratio <= 0.40:
        return _guard_switch(ResolvedLang("en", "evidence", False), prev, stt_lang, stt_confidence)
    # gray zone: an Arabic sentence FRAME (≥2 real Arabic words) carrying
    # English nouns is an Arabic turn; otherwise stay with the conversation.
    if ar_real >= 2:
        return ResolvedLang("ar", "evidence", False)
    if prev:
        return ResolvedLang(prev, "sticky", False)
    return ResolvedLang("ar", "evidence", False)


def _guard_switch(r: ResolvedLang, prev: Optional[str],
                  stt_lang: Optional[str], conf: Optional[float]) -> ResolvedLang:
    """Refuse a language switch that rests on a low-confidence transcript from
    a recognizer that was open in the OTHER language: that is the signature
    of the wrong mic, not of the driver changing language."""
    if prev and r.lang != prev and conf is not None and conf < STT_SWITCH_MIN_CONF:
        return ResolvedLang(prev, "sticky", False, unreliable=True)
    return r


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
