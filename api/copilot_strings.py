"""
api/copilot_strings.py — the backend's localized string catalog (CopilotV2).

Every non-LLM string the BACKEND can put in front of the user lives here, keyed
by the turn's single resolved language. The clients each carry a mirror catalog
(CopilotStrings.kt / CopilotStrings.swift) for client-origin strings; the three
catalogs share a key namespace so tools/check_copilot_parity.py can prove no
surface is missing a language.

Rule enforced by tests/test_copilot_language.py: every key exists in BOTH
languages, and an "ar" value contains Arabic script while an "en" value
contains none.
"""

from typing import Dict

_STRINGS: Dict[str, Dict[str, str]] = {
    # Spoken lead-ins while a slow tool works (perceived-latency cover).
    "lead_search":  {"en": "Let me look that up…",  "ar": "بدوّرلك دلوقتي…"},
    "lead_details": {"en": "One sec, let me check…", "ar": "ثواني، بشوفلك…"},
    # Emitted when a turn produced no text at all.
    "fallback_ok":  {"en": "Okay.",                  "ar": "تمام."},
    # Deterministic translation-pass preamble is never shown; kept for logs.
}

KEYS = frozenset(_STRINGS.keys())


def t(key: str, lang: str) -> str:
    """Catalog lookup — lang is the resolved turn language ('ar'|'en')."""
    entry = _STRINGS[key]
    return entry.get(lang) or entry["en"]
