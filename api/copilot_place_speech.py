"""
api/copilot_place_speech.py — the copilot does not SPEAK Google place content (cost Phase 4; the app repo's
docs/cost/phase4-design.md §6).

Google place content (a place's name, rating, review count, reviews, editorial summary, price, opening hours,
phone, address) reached the model through the place tools, and the model's reply is spoken through TTS and
captioned. With the flag OFF (the default) two gates hold:

  1. the model never RECEIVES the descriptive content — ratings, reviews, summaries, price, open-now, hours,
     phone, address are removed from the place tools' results and the pending action ([for_model]); it keeps the
     places' NAMES and ids only so that "add the second one" / "call it" names the right place to the next tool;
  2. the one output gate (copilot_v2._Emitter.delta) SCRUBS the Google place names the turn has seen — from the
     PLACE tools' results and actions, the pending action and the client's "[Shown: …]" cards — out of the spoken
     text ([scrub]): whole words only (Arabic proclitics و/ب/ف/ل/ال allowed), with their Egyptian-verbalized form,
     their leading segment ("Carrefour - Maadi" → "Carrefour") and the parts around an abbreviation ("Dr. Hamdy
     Clinic" → "Hamdy Clinic"); single generic words ("Total", "Mall", "صيدلية") are never scrubbed. A system rule
     tells the model to refer to places by order and distance.

Not scrubbed, on purpose: road names (route summaries come from Mapbox), what the USER asked for, our own labels
(home / work), and the trip's destination / stops (the driver chose them; the navigation screen shows and speaks
them already). The scrub is a net under the rule, not a guarantee against a paraphrase — a brand said alone when
the stored name is longer, or a transliteration, can still pass (the report's residuals).

The on-screen cards and pins (the show_places / add_stop actions) are unchanged — the driver SEES the places.

    COPILOT_SPEAK_PLACE_CONTENT=1   the previous behaviour (the flag; read at call time)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Set

# Tools whose results carry Google place content.
PLACE_TOOLS = frozenset({"find_places", "place_details", "add_stop", "reroute_via", "change_destination",
                         "call_place", "pending_action"})

# Descriptive Google content: never shown to the model with the flag off.
_CONTENT_KEYS = frozenset({"rating", "reviews", "price", "open_now", "hours", "editorial", "review_samples",
                           "phone", "number", "address", "formatted_address", "user_ratings_total"})

# Keys whose values are GOOGLE place names in a place tool's result / action (never `requested` — the user's own
# words — and never `via` outside reroute_via — elsewhere it is a Mapbox road summary).
_NAME_KEYS = frozenset({"name", "place", "new_destination", "others", "other_candidates", "found_name"})

# Our own labels, generic words and joiners (normalized: no leading ال, ة→ه, ى→ي, أإآ→ا). A name or a variant
# whose EVERY word is in here (or that has no letter at all) is never scrubbed: "Total", "Gas Station",
# «محطة بنزين», "Gate No. 5" → "5".
_GENERIC = frozenset({
    "home", "work", "بيت", "شغل",
    "total", "mobil", "shell", "mall", "cafe", "café", "coffee", "shop", "store", "pharmacy", "hospital", "clinic",
    "bank", "atm", "market", "supermarket", "restaurant", "station", "gas", "fuel", "petrol", "parking", "garage",
    "hotel", "school", "mosque", "church", "center", "centre", "city", "street", "road", "gate", "no", "building",
    "bldg", "branch", "the", "of", "and", "el", "al", "new", "old", "dr", "st", "mr", "mrs", "eng", "co",
    "car", "wash", "bakery", "gym", "kiosk", "laundry", "police", "super", "salon", "barber",
    "مول", "كافيه", "كافي", "قهوه", "صيدليه", "مستشفي", "عياده", "بنك", "سوبرماركت", "ماركت", "مطعم", "محطه",
    "بنزين", "بنزينه", "جراج", "فندق", "مدرسه", "مسجد", "جامع", "كنيسه", "مركز", "شارع", "طريق", "بوابه", "رقم",
    "فرع", "مبني", "مدينه", "جديد", "قديم", "سوبر", "مخبز", "فرن", "كشك", "بقاله", "مغسله", "محل", "ورشه",
    "صالون", "كوافير", "جيم",
})

_LETTER_RX = re.compile(r"[^\W\d_]+")


def _norm_word(w: str) -> str:
    w = w.lower().strip(".")
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")):
        w = w.replace(a, b)
    if w.startswith("ال") and len(w) > 3:
        w = w[2:]
    return w


def _generic(s: str) -> bool:
    words = _LETTER_RX.findall(s)
    return not words or all(_norm_word(w) in _GENERIC for w in words)


_PLACEHOLDER = {"en": "that place", "ar": "المكان ده"}

PLACE_SPEECH_RULE = (
    "SPEAKING PLACES: never say a place's name, rating, reviews, price, opening hours, phone or address — they "
    "come from Google and are shown on the screen, never spoken. Say how many you found and refer to them by "
    "order and distance (\"the first one, 2 km ahead\"); for details, say they're on the screen. You may still "
    "pass a place's name or id to a tool. A yes/no about a place asks about \"the place shown on the screen\"; "
    "if what you found may not be what they asked for, say so and ask them to check the screen first.")

_NOTE = (" Descriptive place content (ratings, reviews, hours, phone, address) is on the screen and is never "
         "spoken; never say place names either — refer to places by order and distance.")

_SHOWN_RX = re.compile(r"\[Shown:\s*([^\]]*)\]")
_SHOWN_ITEM_RX = re.compile(r"(?:^|,)\s*\d+[.)]\s*")
# A proclitic, optionally followed by a tatweel («لـZooba», «الـCity Stars»).
# Stacked like Arabic writes them («ولـZooba», «وبالعزبي», «فلـ…»), a tatweel (one or more, then an optional
# space) after them.
_TATWEEL = r"(?:ـ+\s?)?"
_PROCLITIC = r"(?:(?:[وف]?(?:بال|لل|ال|ب|ل)|و|ف)" + _TATWEEL + r")?"


def place_content_spoken() -> bool:
    return os.getenv("COPILOT_SPEAK_PLACE_CONTENT", "0").strip().lower() in ("1", "true", "yes")


def for_model(tool: str, result: Any) -> Any:
    """The tool result the model may see: unchanged with the flag on or for a tool without place content;
    otherwise without the descriptive content (names and ids stay — a later tool call needs them)."""
    if place_content_spoken() or tool not in PLACE_TOOLS or not isinstance(result, dict):
        return result
    out = _strip(result)
    out["note"] = (str(result.get("note") or "") + _NOTE).strip()
    return out


def _strip(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _strip(x) for k, x in v.items() if k not in _CONTENT_KEYS}
    if isinstance(v, list):
        return [_strip(x) for x in v]
    return v


def _add(out: Set[str], s: Any) -> None:
    if isinstance(s, str):
        s = s.strip()
        if len(s) >= 3 and not _generic(s):
            out.add(s)
    elif isinstance(s, list):
        for x in s:
            _add(out, x)
    elif isinstance(s, dict):
        _add(out, s.get("name"))


def google_names(tool: str, result: Any = None, action: Any = None) -> Set[str]:
    """The Google place names a PLACE tool's result / action carries (empty for any other tool)."""
    out: Set[str] = set()
    if tool not in PLACE_TOOLS:
        return out
    keys = _NAME_KEYS | ({"via"} if tool == "reroute_via" else set())

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys:
                    _add(out, v)
                if isinstance(v, (dict, list)) and k not in keys:
                    walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(result)
    walk(action)
    return out


def names_in_history(messages: Iterable[Dict[str, Any]]) -> Set[str]:
    """The names the client listed as shown ("[Shown: 1. X, 2. Y]") in earlier turns — split on the numbering,
    never on a comma inside a name."""
    out: Set[str] = set()
    for m in messages:
        for grp in _SHOWN_RX.findall(str(m.get("content") or "")):
            for part in _SHOWN_ITEM_RX.split(grp):
                _add(out, part)
    return out


def _variants(name: str, lang: str) -> Set[str]:
    """The name, its leading segment ("Carrefour - Maadi" → "Carrefour"), the parts around an abbreviation
    ("Dr. Hamdy Clinic" → "Hamdy Clinic") and, for Arabic turns, their Egyptian-verbalized forms — never a variant
    without a letter or made only of generic words (a bare "12" would scrub a spoken ETA)."""
    v = {name}
    lead = re.split(r"\s+[-|–—]\s+|\s*[,(]\s*", name)[0].strip()
    if lead != name:
        v.add(lead)
    for part in re.split(r"(?<=\.)\s+", name):
        if part != name:
            v.add(part.strip())
    v = {x for x in v if _usable(x)}
    if lang == "ar":
        try:
            from api.copilot_egy import masri
            v |= {m for m in (masri(x) for x in v) if _usable(m)}
        except Exception:  # pragma: no cover
            pass
    return v


def _usable(x: str) -> bool:
    """A form worth scrubbing: 3+ characters, a real (non-generic) word, not an abbreviation alone ("Dr."), and
    not part of a placeholder (which the emitter's second pass would scrub again)."""
    if len(x) < 3 or _generic(x) or re.fullmatch(r"\w{1,4}\.", x):
        return False
    # dropped only when it matches as a whole word INSIDE a placeholder («مكان», "Place"), not a mere substring
    rx = re.compile(r"(?<!\w)" + _PROCLITIC + _form_rx(x) + r"(?!\w)", re.IGNORECASE)
    return not any(rx.search(ph) for ph in _PLACEHOLDER.values())


_EQUIV = {"ة": "[ةه]", "ه": "[ةه]", "ى": "[ىي]", "ي": "[ىي]", "أ": "[أإآا]", "إ": "[أإآا]", "آ": "[أإآا]",
          "ا": "[أإآا]"}


_END_ONLY = frozenset("ةهىي")


def _form_rx(f: str) -> str:
    """A form as a regex that also matches the common Arabic spelling variants: أ/إ/آ/ا anywhere, ة/ه and ى/ي
    only at the END of a word (where the variation lives — «على» is not «علي»); a one-word form under 4 letters
    is matched exactly."""
    if " " not in f and len(f) < 4:
        return re.escape(f)
    out = []
    for i, c in enumerate(f):
        at_end = i == len(f) - 1 or not f[i + 1].isalpha()
        if c in _END_ONLY and not at_end:
            out.append(re.escape(c))
        else:
            out.append(_EQUIV.get(c, re.escape(c)))
    return "".join(out)


@lru_cache(maxsize=64)
def _pattern(names: frozenset, lang: str):
    """ONE alternation over every form (longest first), so a placeholder is never scrubbed again; a form that
    starts with «ال» also matches after the preposition ل («للعزبي» = ل + العزبي)."""
    alts = []
    for n in names:
        for f in _variants(n, lang):
            alts.append((len(f), _PROCLITIC + _form_rx(f)))
            if f.startswith("ال") and len(f) > 3:
                rest = f[2:]
                alts.append((len(f), r"(?:و|ف)?لل" + _TATWEEL + _form_rx(rest)))
                if rest.startswith("ل"):               # «اللبان» after ل is «للبان», not «لللبان»
                    alts.append((len(f), r"(?:و|ف)?ل" + _TATWEEL + _form_rx(rest)))
    if not alts:
        return None
    alts.sort(key=lambda a: -a[0])
    return re.compile(r"(?<!\w)(?:" + "|".join(a for _, a in alts) + r")(?!\w)", re.IGNORECASE)


def scrub(text: str, names: Iterable[str], lang: str) -> str:
    """`text` with every name (and its variants) replaced by a neutral reference: whole words only, an Arabic
    proclitic allowed in front, one pass (compiled once per set of names)."""
    rx = _pattern(frozenset(n for n in names if n), lang)
    if rx is None:
        return text
    text = rx.sub(_PLACEHOLDER.get(lang, _PLACEHOLDER["en"]), text)
    return re.sub(r"\s{2,}", " ", text).strip()


def rule() -> List[Dict[str, str]]:
    """The system message to add (none with the flag on)."""
    return [] if place_content_spoken() else [{"role": "system", "content": PLACE_SPEECH_RULE}]
