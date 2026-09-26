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
    # Spoken by the client verbatim on an in-stream error — already in the
    # resolved language, so an error can never be the wrong-language reply.
    "err_spoken":   {"en": "Sorry, something went wrong. Try again.",
                     "ar": "معلش، حصلت مشكلة. جرب تاني."},
    "err_busy":     {"en": "One second, I'm a bit busy. Say that again?",
                     "ar": "ثانية واحدة، مشغول شوية. قول تاني؟"},
    "err_no_input": {"en": "I didn't catch that.", "ar": "مسمعتش. قول تاني؟"},
    # The per-user daily limit (api/usage_limits.py) — spoken, in the turn's language.
    "err_daily_limit": {"en": "That's today's limit for Sekka. I'm back tomorrow.",
                        "ar": "ده آخر حد لسِكّة النهارده. هرجع بكرة."},
    # The audio path (api/copilot_stt.py): the ears failed / the day's listening budget is used.
    "err_stt":      {"en": "Sorry, I lost you for a second. Try again?",
                     "ar": "معلش، السمع قطع ثانية. جرّب تاني؟"},
    "err_voice_limit": {"en": "That's today's listening limit for Sekka. I'm back tomorrow.",
                        "ar": "ده آخر حد للسمع النهارده. هرجع بكرة."},
    # A garbled / low-confidence transcript: confirm before acting.
    "unclear_ask":  {"en": "Say that once more?", "ar": "معلش، قولها تاني؟"},
    # Deterministic single-intent commands (no model in the loop): the ack.
    "ack_muted":    {"en": "Guidance muted.",   "ar": "قفلت الصوت."},
    "ack_unmuted":  {"en": "Voice is back on.", "ar": "رجّعت الصوت."},
    "ack_repeat":   {"en": "Sure.",             "ar": "حاضر."},
    "ack_louder":   {"en": "Louder.",           "ar": "علّيت الصوت."},
    "ack_quieter":  {"en": "Quieter.",          "ar": "وطّيت الصوت."},
    "ack_thanks":   {"en": "Anytime.",          "ar": "العفو، أي خدمة."},
    # A sign-off («خلاص كده», "that's all"): the conversation ends here.
    "ack_bye":      {"en": "Okay — I'm here if you need me.", "ar": "تمام، أنا معاك لو احتجت حاجة."},
}

KEYS = frozenset(_STRINGS.keys())


def t(key: str, lang: str) -> str:
    """Catalog lookup — lang is the resolved turn language ('ar'|'en')."""
    entry = _STRINGS[key]
    return entry.get(lang) or entry["en"]
