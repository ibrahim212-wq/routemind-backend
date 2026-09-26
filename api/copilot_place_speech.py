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

# Our own labels and single generic words: never scrubbed.
_NEVER = frozenset({
    "home", "work", "البيت", "الشغل", "بيت", "شغل",
    "total", "mobil", "shell", "mall", "cafe", "café", "coffee", "pharmacy", "hospital", "clinic", "bank", "atm",
    "market", "restaurant", "station", "gas", "fuel", "parking", "hotel", "school", "mosque", "church", "center",
    "centre", "city", "street", "road", "مول", "كافيه", "قهوة", "صيدلية", "مستشفى", "عيادة", "بنك", "سوبرماركت",
    "مطعم", "محطة", "بنزينة", "جراج", "فندق", "مدرسة", "مسجد", "جامع", "كنيسة", "مركز", "شارع", "طريق",
})

_PLACEHOLDER = {"en": "that place", "ar": "المكان ده"}

PLACE_SPEECH_RULE = (
    "SPEAKING PLACES: never say a place's name, rating, reviews, price, opening hours, phone or address — they "
    "come from Google and are shown on the screen, never spoken. Say how many you found and refer to them by "
    "order and distance (\"the first one, 2 km ahead\"); for details, say they're on the screen. You may still "
    "pass a place's name or id to a tool.")

_NOTE = (" Descriptive place content (ratings, reviews, hours, phone, address) is on the screen and is never "
         "spoken; never say place names either — refer to places by order and distance.")

_SHOWN_RX = re.compile(r"\[Shown:\s*([^\]]*)\]")
_SHOWN_ITEM_RX = re.compile(r"(?:^|,)\s*\d+[.)]\s*")
_PROCLITIC = r"(?:وال|بال|فال|لل|ال|و|ب|ف|ل)?"


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
        if len(s) >= 3 and s.lower() not in _NEVER:
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
    v = {name}
    lead = re.split(r"\s+[-|–—]\s+|\s*[,(]\s*", name)[0].strip()
    if lead != name and len(lead.split()) >= 1:
        v.add(lead)
    for part in re.split(r"(?<=\.)\s+", name):           # "Dr. Hamdy Clinic" → "Dr.", "Hamdy Clinic"
        if part != name:
            v.add(part.strip())
    if lang == "ar":
        try:
            from api.copilot_egy import masri
            v |= {masri(x) for x in list(v)}
        except Exception:  # pragma: no cover
            pass
    return {x for x in v if len(x) >= 3 and x.lower() not in _NEVER and not re.fullmatch(r"\w{1,3}\.", x)}


def scrub(text: str, names: Iterable[str], lang: str) -> str:
    """`text` with every name (and its variants) replaced by a neutral reference: whole words only, an Arabic
    proclitic allowed in front, longest first, case-insensitive."""
    ph = _PLACEHOLDER.get(lang, _PLACEHOLDER["en"])
    forms: Set[str] = set()
    for n in set(names):
        forms |= _variants(n, lang)
    for f in sorted(forms, key=len, reverse=True):
        rx = r"(?<!\w)" + _PROCLITIC + re.escape(f) + r"(?!\w)"
        text = re.sub(rx, ph, text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def rule() -> List[Dict[str, str]]:
    """The system message to add (none with the flag on)."""
    return [] if place_content_spoken() else [{"role": "system", "content": PLACE_SPEECH_RULE}]
