"""
api/copilot_place_speech.py — the copilot never SPEAKS Google place content (cost Phase 4; the app repo's
docs/cost/phase4-design.md §6).

Google place content (a place's name, rating, review count, reviews, editorial summary, price, opening hours,
phone, address) reached the model through the place tools and the trip context, and the model's reply is spoken
through TTS and captioned. With the flag OFF (the default) two gates hold:

  1. the model never RECEIVES the descriptive content — ratings, reviews, summaries, price, open-now, hours,
     phone, address are removed from the place tools' results ([for_model]); it keeps the places' NAMES and ids
     only so that "add the second one" / "call it" can still name the right place to the next tool;
  2. the one output gate (copilot_v2._Emitter.delta) SCRUBS every Google place name the turn has seen — from the
     tools' results and actions, the trip context (destination, stops, alternatives), the pending action and the
     client's "[Shown: …]" history — out of the spoken text ([scrub]), so a name the model says anyway is never
     spoken; a system rule tells the model to refer to places by order and distance.

The on-screen cards and pins (the show_places / add_stop actions) are unchanged — the driver SEES the places.

    COPILOT_SPEAK_PLACE_CONTENT=1   the previous behaviour (the flag; read at call time)
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Set

# Tools whose results carry Google place content.
PLACE_TOOLS = frozenset({"find_places", "place_details", "add_stop", "reroute_via", "change_destination",
                         "call_place", "skip_next_stop", "pending_action"})

# Descriptive Google content: never shown to the model with the flag off.
_CONTENT_KEYS = frozenset({"rating", "reviews", "price", "open_now", "hours", "editorial", "review_samples",
                           "phone", "number", "address", "formatted_address", "user_ratings_total"})

# Keys whose string values are place NAMES (collected for the output scrub).
_NAME_KEYS = frozenset({"name", "place", "via", "new_destination", "requested", "others", "other_candidates",
                        "skipped", "match", "dest_name", "destination_name", "leg_to", "found_name", "place_name",
                        "query_name", "title"})

# Our own labels, never scrubbed.
_OWN_LABELS = frozenset({"home", "work", "البيت", "الشغل", "بيت", "شغل"})

_PLACEHOLDER = {"en": "that place", "ar": "المكان ده"}

PLACE_SPEECH_RULE = (
    "SPEAKING PLACES: never say a place's name, rating, reviews, price, opening hours, phone or address — they "
    "come from Google and are shown on the screen, never spoken. Say how many you found and refer to them by "
    "order and distance (\"the first one, 2 km ahead\"); for details, say they're on the screen. You may still "
    "pass a place's name or id to a tool.")

_NOTE = (" Descriptive place content (ratings, reviews, hours, phone, address) is on the screen and is never "
         "spoken; never say place names either — refer to places by order and distance.")

_SHOWN_RX = re.compile(r"\[Shown:\s*([^\]]*)\]")


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


def names_in(obj: Any, out: Set[str] | None = None) -> Set[str]:
    """Every string under a name-like key, anywhere in `obj` (dicts / lists), except our own labels."""
    out = set() if out is None else out

    def add(s: Any) -> None:
        if isinstance(s, str):
            s = s.strip()
            if len(s) >= 3 and s.lower() not in _OWN_LABELS:
                out.add(s)
        elif isinstance(s, list):
            for x in s:
                add(x)

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _NAME_KEYS:
                add(v)
            if isinstance(v, (dict, list)):
                names_in(v, out)
    elif isinstance(obj, list):
        for x in obj:
            names_in(x, out)
    return out


def names_in_history(messages: Iterable[Dict[str, Any]]) -> Set[str]:
    """The names the client listed as shown ("[Shown: 1. X, 2. Y]") in earlier turns."""
    out: Set[str] = set()
    for m in messages:
        for grp in _SHOWN_RX.findall(str(m.get("content") or "")):
            for part in grp.split(","):
                s = re.sub(r"^\s*\d+[.)]\s*", "", part).strip()
                if len(s) >= 3 and s.lower() not in _OWN_LABELS:
                    out.add(s)
    return out


def scrub(text: str, names: Iterable[str], lang: str) -> str:
    """`text` with every name replaced by a neutral reference (longest names first, case-insensitive)."""
    ph = _PLACEHOLDER.get(lang, _PLACEHOLDER["en"])
    for n in sorted(set(names), key=len, reverse=True):
        if n and n.lower() in text.lower():
            text = re.sub(re.escape(n), ph, text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def context_names(ctx: Dict[str, Any]) -> Set[str]:
    """The Google names the trip context carries (destination, stops, alternatives' legs)."""
    return names_in(ctx if isinstance(ctx, dict) else {})


def rule() -> List[Dict[str, str]]:
    """The system message to add (none with the flag on)."""
    return [] if place_content_spoken() else [{"role": "system", "content": PLACE_SPEECH_RULE}]
