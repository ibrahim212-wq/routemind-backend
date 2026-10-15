"""Text normalization for the place index — ONE spec, four copies.

This is the reference. android/.../places/PlaceText.kt, ios/Runner/Places/PlaceText.swift and the
pipeline itself use exactly these tables and steps; tool/place_search_parity.py replays a fixture of
real Egyptian names through every copy and fails on a single differing code point. The index is
built with this file, so a client that normalizes differently simply cannot find what is there.

It extends lib/features/search/domain/text_normalizer.dart (which keeps doing query understanding
on the phone) with what an index needs and the Dart side never had to decide:

  * a closed definition of a word character — no Unicode category tables, which differ between
    Python, the JVM and Swift by Unicode version;
  * ڤ پ چ folded to ف ب ج: Egyptian Facebook pages write «ڤودافون» and «چيلاتي», people type
    «فودافون» and «جيلاتي»;
  * Arabic letters and digits split apart («شارع9» = «شارع 9»), Latin left whole because a digit
    inside a Latin word is a Franco letter («m3ady», «2al3a»);
  * the attached-preposition forms an index needs as variants (للسيارات → سيارات, بالمعادي →
    معادي) besides the article.

THE STEPS (normalize)
  1  lower-case ASCII A-Z; fold the Latin letters in LATIN_FOLD
  2  drop tashkeel (U+064B..U+065F, U+0670, U+06D6..U+06ED) and tatweel (U+0640)
  3  unify Arabic letters by ARABIC_UNIFY (alef forms → ا, ى ئ ی → ي, ة → ه, ؤ → و, ء dropped, …)
  4  Arabic-Indic / Extended Arabic-Indic digits → ASCII
  5  every character that is not a WORD character becomes a space (is_word_char)
  6  a space between an Arabic letter and a digit, either order
  7  collapse runs of spaces, trim

THE SKELETON (cross-script / Franco matching) is the Dart Transliteration's consonant skeleton,
per word: both scripts project onto one consonant alphabet (ش and sh → c, خ and kh/5 → x,
ح and 7 → 7, غ and gh/8 → 8, ع/3 and vowels dropped), so «مستشفى» and «mostashfa» are both
"mstcf". Latin words branch on the ambiguous Franco characters ('2' = hamza or Cairene qaf, a final 't' = construct-state ة or a real t), at most
MAX_SKELETON_VARIANTS. Unlike the Dart query-side skeleton, ق ك q k are ONE symbol here: Egyptian
Latin spellings write ق as k («Boulak», «Kasr», «Dokki»), and a place index cannot branch on it.
"""
from __future__ import annotations

from functools import lru_cache

NORM_VERSION = 1

LATIN_FOLD = {
    "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a", "ā": "a",
    "ç": "c", "ć": "c", "č": "c",
    "é": "e", "è": "e", "ê": "e", "ë": "e", "ē": "e", "ę": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i", "ī": "i", "ı": "i",
    "ñ": "n", "ń": "n",
    "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o", "ō": "o", "ø": "o",
    "ś": "s", "š": "s", "ş": "s", "ß": "ss",
    "ú": "u", "ù": "u", "û": "u", "ü": "u", "ū": "u",
    "ý": "y", "ÿ": "y",
    "ź": "z", "ż": "z", "ž": "z",
    "ğ": "g",
    # upper-case forms that lower() alone would leave to the platform's tables
    "Á": "a", "À": "a", "Â": "a", "Ä": "a", "Ã": "a", "Å": "a", "Ā": "a",
    "Ç": "c", "É": "e", "È": "e", "Ê": "e", "Ë": "e", "Í": "i", "Î": "i", "Ï": "i", "İ": "i",
    "Ñ": "n", "Ó": "o", "Ô": "o", "Ö": "o", "Ø": "o", "Ş": "s", "Š": "s",
    "Ú": "u", "Û": "u", "Ü": "u", "Ý": "y", "Ž": "z", "Ğ": "g",
}

# step 3 — Arabic letter unification. Value "" = drop.
ARABIC_UNIFY = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
    "ى": "ي", "ئ": "ي", "ی": "ي", "ې": "ي", "ۍ": "ي",
    "ة": "ه", "ۀ": "ه", "ہ": "ه", "ھ": "ه",
    "ؤ": "و",
    "ء": "",
    "گ": "ك", "ک": "ك",
    "ڤ": "ف", "ڨ": "ف",
    "پ": "ب",
    "چ": "ج",
}

def _is_tashkeel(cp: int) -> bool:
    return 0x064B <= cp <= 0x065F or cp == 0x0670 or 0x06D6 <= cp <= 0x06ED or cp == 0x0640


def is_word_char(ch: str) -> bool:
    """Step 5's closed definition of a character that belongs to a word."""
    cp = ord(ch)
    if 0x61 <= cp <= 0x7A or 0x30 <= cp <= 0x39:      # a-z 0-9
        return True
    if 0x0621 <= cp <= 0x063A or 0x0641 <= cp <= 0x064A:  # Arabic letters
        return True
    if 0x0671 <= cp <= 0x06D3:                          # extended Arabic letters (after unify)
        return True
    if 0x00DF <= cp <= 0x024F and cp != 0x00F7 and cp != 0x00D7:  # Latin extended, not × ÷
        return True
    if 0x0370 <= cp <= 0x03FF or 0x0400 <= cp <= 0x04FF:  # Greek, Cyrillic
        return True
    return False


def is_arabic_letter(ch: str) -> bool:
    cp = ord(ch)
    return 0x0621 <= cp <= 0x064A or 0x0671 <= cp <= 0x06D3


def is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


@lru_cache(maxsize=200_000)
def normalize(text: str | None) -> str:
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        # 1 — case + Latin folding
        if 0x41 <= cp <= 0x5A:
            ch = chr(cp + 32)
        elif ch in LATIN_FOLD:
            out.append(LATIN_FOLD[ch])
            continue
        # 2 — decoration
        if _is_tashkeel(cp):
            continue
        # 3 — Arabic unification
        if ch in ARABIC_UNIFY:
            rep = ARABIC_UNIFY[ch]
            if rep:
                out.append(rep)
            continue
        # 4 — digits
        if 0x0660 <= cp <= 0x0669:
            out.append(chr(0x30 + cp - 0x0660))
            continue
        if 0x06F0 <= cp <= 0x06F9:
            out.append(chr(0x30 + cp - 0x06F0))
            continue
        # 5 — everything that is not a word character is a separator
        out.append(ch if is_word_char(ch) else " ")
    # 6 — split Arabic letters from digits
    s = "".join(out)
    res: list[str] = []
    prev = ""
    for ch in s:
        if prev and ch != " " and prev != " ":
            if (is_arabic_letter(prev) and is_digit(ch)) or (is_digit(prev) and is_arabic_letter(ch)):
                res.append(" ")
        res.append(ch)
        prev = ch
    # 7 — collapse
    return " ".join("".join(res).split())


def tokens(text: str | None) -> list[str]:
    n = normalize(text)
    return n.split(" ") if n else []


def script_of(text: str) -> str:
    """'ar' | 'latin' | 'mixed' | 'neutral' — of the NORMALIZED text."""
    ar = la = 0
    for ch in text:
        if is_arabic_letter(ch):
            ar += 1
        elif "a" <= ch <= "z":
            la += 1
    if ar == 0 and la == 0:
        return "neutral"
    if ar and la:
        tot = ar + la
        if ar / tot > 0.8:
            return "ar"
        if la / tot > 0.8:
            return "latin"
        return "mixed"
    return "ar" if ar else "latin"


def is_arabic_token(tok: str) -> bool:
    return any(is_arabic_letter(c) for c in tok)


# Attached Arabic prefixes an index adds as variants. Ordered longest first; each needs the
# remaining stem to keep MIN_STEM letters, so «الف» never becomes «ف».
ARABIC_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال")
MIN_STEM = 3
LATIN_ARTICLES = ("el", "al")
# Latin words that carry no meaning in a place name and are never indexed alone.
LATIN_STOP = frozenset({"el", "al", "the", "of", "and", "de", "la", "le", "les", "du", "des", "for", "in", "at", "&"})
ARABIC_STOP = frozenset({"و", "في", "فى", "من", "على", "علي", "عند", "ال"})


@lru_cache(maxsize=400_000)
def stems(tok: str) -> list[str]:
    """(memoized — callers never mutate the list) The token itself and its article / attached-preposition-free forms (no duplicates)."""
    out = [tok]
    if is_arabic_token(tok):
        for p in ARABIC_PREFIXES:
            if tok.startswith(p) and len(tok) - len(p) >= MIN_STEM:
                stem = tok[len(p):]
                if stem not in out:
                    out.append(stem)
                break
    else:
        for a in LATIN_ARTICLES:
            if tok.startswith(a) and len(tok) > 4 and tok not in ("alex", "alexandria"):
                stem = tok[len(a):]
                if stem not in out:
                    out.append(stem)
                break
    return out


def is_stopword(tok: str) -> bool:
    return tok in LATIN_STOP or tok in ARABIC_STOP


# ── skeleton ────────────────────────────────────────────────────────────────

MAX_SKELETON_VARIANTS = 4

ARABIC_WEAK = frozenset("اويهةىءأإآؤئع")
ARABIC_TO_SKEL = {
    "ب": "b", "ت": "t", "ث": "t", "ج": "g", "ح": "7", "خ": "x", "د": "d", "ذ": "z", "ر": "r",
    "ز": "z", "س": "s", "ش": "c", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "غ": "8", "ف": "f",
    "ق": "k", "ك": "k", "ل": "l", "م": "m", "ن": "n", "پ": "b", "ڤ": "f", "چ": "g",
}
LATIN_DIGRAPHS = {
    "sh": "c", "ch": "c", "kh": "x", "gh": "8", "th": "t", "dh": "z", "ph": "f", "ck": "k",
    "ss": "s", "tt": "t", "ll": "l", "mm": "m", "nn": "n", "bb": "b", "dd": "d", "rr": "r",
    "ee": "", "oo": "", "aa": "", "ou": "",
}
LATIN_TO_SKEL = {
    "a": "", "e": "", "i": "", "o": "", "u": "", "b": "b", "c": "k", "d": "d", "f": "f", "g": "g",
    "h": "", "j": "g", "k": "k", "l": "l", "m": "m", "n": "n", "p": "b", "q": "k", "r": "r",
    "s": "s", "t": "t", "v": "f", "w": "", "x": "ks", "y": "", "z": "z",
    "3": "", "5": "x", "6": "t", "7": "7", "8": "8", "9": "s", "4": "c",
    "0": "", "1": "", "2": "",
}
LATIN_AMBIGUOUS = {"2": ("", "k")}


def _skel_arabic(word: str) -> str:
    buf: list[str] = []
    last = ""
    for i, ch in enumerate(word):
        if i == 0 and ch in ("و", "ي"):
            sym = "w" if ch == "و" else "y"
            buf.append(sym)
            last = sym
            continue
        if ch in ARABIC_WEAK:
            continue
        sym = ARABIC_TO_SKEL.get(ch, "")
        if not sym or sym == last:
            continue
        buf.append(sym)
        last = sym
    return "".join(buf)


def _skel_latin(word: str) -> list[str]:
    w = word
    if len(w) > 4 and (w.startswith("el") or w.startswith("al")):
        w = w[2:]
    dropped_final_t = len(w) > 3 and w.endswith("t") and not w.endswith("tt")
    variants = [""]
    i = 0
    n = len(w)
    while i < n:
        is_last = i == n - 1
        options: tuple[str, ...] | None = None
        consumed = 1
        if i + 1 < n and w[i:i + 2] in LATIN_DIGRAPHS:
            options = (LATIN_DIGRAPHS[w[i:i + 2]],)
            consumed = 2
        else:
            ch = w[i]
            if is_last and ch == "t" and dropped_final_t:
                options = ("", "t")
            elif ch in ("y", "w") and i == 0:
                options = (ch,)
            elif ch in LATIN_AMBIGUOUS:
                options = LATIN_AMBIGUOUS[ch]
            else:
                options = (LATIN_TO_SKEL.get(ch, ""),)
        nxt: list[str] = []
        for prefix in variants:
            for opt in options:
                if not opt:
                    cand = prefix
                elif prefix.endswith(opt):
                    cand = prefix
                else:
                    cand = prefix + opt
                if cand not in nxt:
                    nxt.append(cand)
        variants = nxt[:MAX_SKELETON_VARIANTS]
        i += consumed
    out = [v for v in variants if v]
    return out or [""]


@lru_cache(maxsize=400_000)
def skeletons(tok: str) -> list[str]:
    """(memoized) Consonant skeleton(s) of ONE normalized token. Arabic → exactly one; Latin → 1..4."""
    if not tok:
        return [""]
    if is_arabic_token(tok):
        # an Arabic token loses its article first, as the Dart side's normalizeLoose does
        for p in ARABIC_PREFIXES:
            if tok.startswith(p) and len(tok) - len(p) >= MIN_STEM:
                tok = tok[len(p):]
                break
        return [_skel_arabic(tok)]
    if all(is_digit(c) for c in tok):
        return [tok]
    return _skel_latin(tok)


# ── loose form ──────────────────────────────────────────────────────────────

_LOOSE_DROP_AR = frozenset("اويهى")
_LOOSE_DROP_LA = frozenset("aeiouyhw")


@lru_cache(maxsize=400_000)
def loose(tok: str) -> str:
    """The token without its long vowels / weak letters after the first character, doubled letters
    collapsed: «بيشوي» = «بشوي» = "بش", «الحاج» → «حاج» = «حج», "abou" = "abu" = "ab". Spelling
    drift in Egyptian names is mostly here, and an edit distance on 4-letter words is too blunt."""
    st = stems(tok)
    t = st[-1] if len(st) > 1 else tok
    drop = _LOOSE_DROP_AR if is_arabic_token(t) else _LOOSE_DROP_LA
    out = []
    for i, ch in enumerate(t):
        if i > 0 and ch in drop:
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


# ── edit distance ───────────────────────────────────────────────────────────

def damerau(a: str, b: str, ceiling: int) -> int:
    """Optimal-string-alignment distance, early exit above `ceiling` (returns ceiling + 1)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > ceiling:
        return ceiling + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev2 = [0] * (lb + 1)
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        row_min = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > ceiling:
            return ceiling + 1
        prev2, prev, cur = prev, cur, prev2
    return prev[lb] if prev[lb] <= ceiling else ceiling + 1


def max_edits(tok: str) -> int:
    """Typo budget by token length — one edit from 4 letters, two from 8 (Arabic one letter sooner)."""
    n = len(tok)
    if is_arabic_token(tok):
        return 0 if n < 3 else (1 if n < 7 else 2)
    return 0 if n < 4 else (1 if n < 8 else 2)
