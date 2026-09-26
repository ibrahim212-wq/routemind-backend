"""
api/copilot_place_speech.py — the copilot never SPEAKS Google place content (cost Phase 4; the app repo's
docs/cost/phase4-design.md §6).

Google place content (a place's name, rating, review count, reviews, editorial summary, price, opening hours,
phone, address) reached the model through the place tools and the trip context, and the model's reply is spoken
through TTS and captioned. With the flag OFF (the default) the model never RECEIVES that content, so it cannot say
it: the tool results keep only what is ours (how many, the order, our distance / detour, the ids a later tool call
needs) and a system rule tells it the places are on the screen. The on-screen cards and pins (the show_places /
add_stop actions) still carry everything — the driver SEES the places; the voice refers to them by order and
distance.

    COPILOT_SPEAK_PLACE_CONTENT=1   the previous behaviour (the flag; read at call time)
"""
from __future__ import annotations

import os
from typing import Any, Dict

# Tools whose results carry Google place content (find_places / place_details / add_stop / reroute_via from
# Places; change_destination from a Places search; call_place from Place Details).
PLACE_TOOLS = frozenset({"find_places", "place_details", "add_stop", "reroute_via", "change_destination",
                         "call_place"})

# Fields that are Google content wherever they appear in those results.
_CONTENT_KEYS = frozenset({"name", "place", "via", "new_destination", "rating", "reviews", "price", "open_now",
                           "hours", "editorial", "review_samples", "phone", "number", "address", "requested",
                           "others", "match"})

# Fields that are ours (or an id a later call needs) and stay.
_KEEP_IN_PLACES = ("id", "detour_min", "away")

PLACE_SPEECH_RULE = (
    "SPEAKING PLACES: never say a place's name, rating, reviews, price, opening hours, phone or address — they "
    "come from Google and are shown on the screen, never spoken. Say how many you found and refer to them by "
    "order and distance (\"the first one, 2 km ahead\"); for details, say they're on the screen.")

_NOTE = (" Place content (names, ratings, reviews, hours, phone, address) is on the screen and is never spoken: "
         "refer to places by order and distance.")


def place_content_spoken() -> bool:
    return os.getenv("COPILOT_SPEAK_PLACE_CONTENT", "0").strip().lower() in ("1", "true", "yes")


def for_model(tool: str, result: Any) -> Any:
    """The tool result the model may see: unchanged with the flag on or for a tool without place content;
    otherwise without any Google place content."""
    if place_content_spoken() or tool not in PLACE_TOOLS or not isinstance(result, dict):
        return result
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if k in _CONTENT_KEYS:
            continue
        if k == "places" and isinstance(v, list):
            out["places"] = [
                dict({"n": i + 1}, **{kk: p[kk] for kk in _KEEP_IN_PLACES if isinstance(p, dict) and kk in p})
                for i, p in enumerate(v)]
            continue
        out[k] = v
    out["note"] = (str(result.get("note") or "") + _NOTE).strip()
    return out


def context_for_model(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The trip context without the destination's / stops' names (the user picked them from Google search)."""
    if place_content_spoken() or not isinstance(ctx, dict):
        return ctx
    c = dict(ctx)
    for k in ("dest_name", "destination_name"):
        if c.get(k):
            c[k] = "set (shown on the screen)"
    if isinstance(c.get("stops"), list):
        c["stops"] = [f"stop {i + 1}" for i in range(len(c["stops"]))]
    return c
