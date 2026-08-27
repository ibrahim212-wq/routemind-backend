"""
api/copilot_fastpath.py — deterministic trip-fact answers (CopilotV2).

For the questions a driver asks most ("كام رادار قدامي؟", "how long left?"),
the trip context ALREADY holds the exact answer. Routing those through an LLM
adds latency and a nonzero chance of a wrong number — the worst failure this
feature can have. This module answers them deterministically: the spoken value
IS the context value, formatted by the same speech-number rules the prompt
teaches. tests/test_copilot_fastpath.py asserts spoken == source exactly.

Guardrails (precision over recall — a miss just falls through to the LLM):
  • only when no action is pending and the utterance is short (≤ 9 words)
  • never when the text carries action/POI vocabulary (add, stop, بنزينة …)
  • never when TWO intents match (multi-intent goes to the model)
  • a required context field missing → no fast-path (the LLM answers honestly)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from api.copilot_lang import speak_minutes, speak_distance

# ── Guards ────────────────────────────────────────────────────────────────────
# Any of these words → the turn wants an ACTION or a place — never fast-path.
_ACTION_WORDS = re.compile(
    r"(add|stop\b|station|pharmacy|atm|restaurant|cafe|coffee|mosque|hospital|"
    r"parking|switch|change|avoid|reroute|route to|take me|go to|navigate|"
    r"home\b|work\b|mute|unmute|volume|report|share|remind|call|cancel|"
    r"بنزينه|بنزينة|محطه|محطة|صيدليه|صيدلية|مطعم|كافيه|قهوه|قهوة|جامع|مسجد|"
    r"مستشفى|موقف|جراج|ضيف|اضف|زود|حول|بدل|غير|شيل|الغي|ألغي|وقف|روح|خدني|"
    r"وديني|اقفل|افتح|اعمل|بلغ|ابعت|فكرني|اتصل|كلم)")


def _too_actiony(t: str) -> bool:
    return bool(_ACTION_WORDS.search(t))


# ── Intent patterns (each entry: id, compiled regexes, required ctx keys) ─────
def _rx(*pats: str):
    return [re.compile(p) for p in pats]


_INTENTS = [
    ("cameras_count", _rx(
        r"how many (speed )?(camera|cameras|radar|radars)",
        r"(كام|عدد) (رادار|كاميرا|الرادارات|الكاميرات)",
        r"(رادار|كاميرا)\S* (كام|قد ايه)",
        r"(في|فيه|فى) (رادارات|كاميرات|رادار)")),
    ("next_camera", _rx(
        r"(where|how far).{0,12}(next )?(camera|radar)",
        r"(next|nearest|closest) (camera|radar)",
        r"(فين|بعد كام|علي بعد كام|على بعد كام).{0,10}(رادار|كاميرا)",
        r"(الرادار|الكاميرا) (الجاي|الجايه|الجاية|القادم|فين)")),
    ("speed_limit", _rx(
        r"(what('|’)?s|whats|what is) the speed limit",
        r"^speed limit\??$",
        r"(السرعه|السرعة) (القصوى|المسموحه|المسموحة|المسموح)",
        r"حد السرعه|حد السرعة|اقصى سرعه|أقصى سرعة")),
    ("current_speed", _rx(
        r"(how fast am i|what('|’)?s my speed|whats my speed|what is my speed)",
        r"(سرعتي|سرعتى) كام", r"ماشي بكام|ماشى بكام")),
    ("current_road", _rx(
        r"(what|which) (road|street) (am i on|is this)",
        r"(انا|إحنا|احنا) (في|فى|علي|على) (شارع|طريق) (ايه|إيه|مين)",
        r"(الشارع|الطريق) (ده|دا) (اسمه ايه|ايه|إيه)",
        r"(ماشي|ماشى|ماشيين) (في|فى|علي|على) (ايه|إيه|فين)")),
    ("next_turn", _rx(
        r"(what('|’)?s|whats|what is) (my |the )?next (turn|maneuver|exit)",
        r"where do i turn", r"^next turn\??$",
        r"(الف|ألف|هلف|احول|أحول|اخش|أخش) فين",
        r"(اللفه|اللفة|الحركه|الحركة|الخروجه|الخروجة|المخرج) (الجايه|الجاية|الجاي|القادمه|القادمة|فين)",
        r"بعد كام (الف|ألف|لفه|لفة)")),
    ("arrival_time", _rx(
        r"(what time|when) (will|do) (i|we) (arrive|get there|reach)",
        r"arrival time",
        r"(هوصل|هنوصل|نوصل) (الساعه|الساعة) كام",
        r"(هوصل|هنوصل) امتي|(هوصل|هنوصل) امتى|(هوصل|هنوصل) إمتى")),
    ("eta_remaining", _rx(
        r"how (long|much longer|much time)( is)?( left| remaining| to go)?\??$",
        r"how long (till|until|before) (i|we) (arrive|get)",
        r"(time|minutes) (left|remaining)", r"^(what('|’)?s the |whats the )?eta\??$",
        r"(فاضل|باقي|باقى|فاضلي|فاضللي) (كام|قد ايه|اد ايه)(?! (كيلو|كم))",
        r"(كام|قد ايه) (فاضل|باقي|باقى)(?! (كيلو|كم))",
        r"الوقت (المتبقي|المتبقى|الباقي|الباقى)")),
    ("distance_remaining", _rx(
        r"how far( left| to go| is it| remaining)?\??$",
        r"(distance|km|kilometers) (left|remaining|to go)",
        r"(فاضل|باقي|باقى) كام (كيلو|كم)", r"كام كيلو (فاضل|باقي|باقى)",
        r"(المسافه|المسافة) (المتبقيه|المتبقية|الباقيه|الباقية|كام)")),
]

# eta/distance patterns must not swallow traffic questions ("how long is the
# jam") or camera ones — those carry their own vocabulary; block generics.
_TRAFFIC_WORDS = re.compile(r"(traffic|jam|congestion|زحمه|زحمة|زحام|عربيات)")
_GENERIC_BLOCK = {"eta_remaining", "distance_remaining", "arrival_time"}


def match_intent(text: str) -> Optional[str]:
    """The single matching fast-path intent, or None (ambiguous/actiony/none)."""
    t = " ".join((text or "").lower().split())
    if not t or len(t.split()) > 9:
        return None
    if _too_actiony(t):
        return None
    hits = []
    for intent, regexes in _INTENTS:
        if any(r.search(t) for r in regexes):
            if intent in _GENERIC_BLOCK and _TRAFFIC_WORDS.search(t):
                continue
            hits.append(intent)
    if len(hits) != 1:
        return None
    return hits[0]


# ── Answer templates (speech-natural, numbers exactly from ctx) ───────────────
def _clock_after(local_time: str, minutes: int) -> Optional[str]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", (local_time or "").strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    total = (h * 60 + mi + int(minutes)) % (24 * 60)
    return f"{total // 60}:{total % 60:02d}"


def _radar_count_ar(n: int) -> str:
    if n == 1:
        return "رادار واحد"
    if n == 2:
        return "رادارين"
    if n <= 10:
        return f"{n} رادارات"
    return f"{n} رادار"


def answer(intent: str, ctx: Dict[str, Any], lang: str) -> Optional[str]:
    """Deterministic spoken answer, or None when the context can't back it
    (missing data falls through to the LLM, which answers honestly)."""
    ar = lang == "ar"

    if intent == "cameras_count":
        cams = ctx.get("cameras_ahead")
        if cams is None:
            return None
        n = len(cams)
        if n == 0:
            return ("مفيش رادارات على باقي طريقك." if ar
                    else "No speed cameras on the rest of your route.")
        first = cams[0] if isinstance(cams[0], dict) else {}
        d = first.get("distance_ahead_km", first.get("distance_km"))
        dist = speak_distance(float(d), lang) if d is not None else None
        lim = first.get("limit_kmh", first.get("limit"))
        if ar:
            s = f"قدامك {_radar_count_ar(n)}"
            if dist:
                s += f" — أقربهم بعد {dist} تقريبًا"
            if lim:
                s += f"، والسرعة عنده {int(lim)}"
            return s + "."
        s = f"{n} camera{'s' if n != 1 else ''} ahead"
        if dist:
            s += f" — the nearest in about {dist}"
        if lim:
            s += f", limit {int(lim)}"
        return s + "."

    if intent == "next_camera":
        cams = ctx.get("cameras_ahead")
        if cams is None:
            return None
        if not cams:
            return ("مفيش رادارات قدامك على الطريق." if ar
                    else "No cameras ahead on your route.")
        first = cams[0] if isinstance(cams[0], dict) else {}
        d = first.get("distance_ahead_km", first.get("distance_km"))
        if d is None:
            return None
        dist = speak_distance(float(d), lang)
        lim = first.get("limit_kmh", first.get("limit"))
        if ar:
            s = f"الرادار الجاي بعد {dist} تقريبًا"
            if lim:
                s += f"، والسرعة عنده {int(lim)}"
            return s + "."
        s = f"Next camera in about {dist}"
        if lim:
            s += f", limit {int(lim)}"
        return s + "."

    if intent == "speed_limit":
        lim = ctx.get("speed_limit_kmh")
        if lim is None:
            return ("مفيش بيانات حد سرعة للطريق ده." if ar
                    else "No posted speed-limit data for this road.")
        spd = ctx.get("speed_kmh")
        over = spd is not None and float(spd) > float(lim) + 3
        if ar:
            s = f"السرعة القصوى هنا {int(lim)}"
            if over:
                s += " — وانت عديها شوية، خد بالك"
            return s + "."
        s = f"The limit here is {int(lim)}"
        if over:
            s += " — you're a bit over it"
        return s + "."

    if intent == "current_speed":
        spd = ctx.get("speed_kmh")
        if spd is None:
            return None
        v = int(round(float(spd)))
        return (f"سرعتك حوالي {v} كيلومتر في الساعة." if ar
                else f"You're doing about {v} kilometers per hour.")

    if intent == "current_road":
        road = ctx.get("current_road")
        if not road:
            return None
        return (f"انت ماشي على {road}." if ar else f"You're on {road}.")

    if intent == "next_turn":
        man = ctx.get("next_maneuver")
        if not man:
            return None
        d = ctx.get("next_maneuver_m")
        if d is not None:
            dist = speak_distance(float(d) / 1000.0, lang)
            return (f"بعد {dist} تقريبًا: {man}." if ar
                    else f"In about {dist}: {man}.")
        return (f"الحركة الجاية: {man}." if ar else f"Next up: {man}.")

    if intent == "arrival_time":
        eta = ctx.get("eta_min", ctx.get("remaining_time_min"))
        if eta is None:
            return None
        clock = _clock_after(ctx.get("local_time") or "", int(eta))
        if clock:
            return (f"هتوصل حوالي الساعة {clock}." if ar
                    else f"You'll arrive around {clock}.")
        return (f"فاضل {speak_minutes(int(eta), lang)} تقريبًا." if ar
                else f"About {speak_minutes(int(eta), lang)} to go.")

    if intent == "eta_remaining":
        eta = ctx.get("eta_min", ctx.get("remaining_time_min"))
        if eta is None:
            return None
        mins = speak_minutes(int(eta), lang)
        clock = _clock_after(ctx.get("local_time") or "", int(eta))
        if ar:
            s = f"فاضل {mins} تقريبًا"
            if clock:
                s += f"، هتوصل حوالي {clock}"
            return s + "."
        s = f"About {mins} to go"
        if clock:
            s += f", arriving around {clock}"
        return s + "."

    if intent == "distance_remaining":
        km = ctx.get("remaining_km", ctx.get("remaining_distance_km"))
        if km is None:
            return None
        dist = speak_distance(float(km), lang)
        return (f"فاضل {dist} تقريبًا." if ar else f"About {dist} to go.")

    return None


def try_fastpath(text: str, ctx: Dict[str, Any], lang: str,
                 has_pending_action: bool) -> Optional[str]:
    """The one entry point: spoken answer or None."""
    if has_pending_action or not ctx:
        return None
    intent = match_intent(text)
    if not intent:
        return None
    return answer(intent, ctx, lang)
