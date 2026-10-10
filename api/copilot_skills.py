"""
api/copilot_skills.py — what Sekka can work out on the server that no phone or chat model can (2026-10-10).

Three skills, each a tool the model calls and the server answers from data — no phone executor, so they work
on the phone, on Android Auto and on CarPlay alike, and cost nothing but the tool round:

  plan_departure  "When should I head back?" / «أرجع امتى عشان ما ألبسش في الزحمة؟» — RouteMind's own
                  prediction model (the per-junction day-of-week × hour baseline the GCN-LSTM-Prophet pipeline
                  was trained on, model/tier1.py) run over THIS route or the trip back, slot by slot, for the
                  next hours or tomorrow. A chat assistant can only recite folklore about rush hour; this is the
                  route's own learned curve. Honest by construction: below the model's junction coverage
                  (outside Cairo & Giza) it says it cannot predict that road.
  prayer_times    the five prayers where the car IS, by the Egyptian General Authority of Survey method (Fajr
                  19.5°, Isha 17.5°, Asr shadow 1 — what every Egyptian mosque calendar prints), in the phone's
                  own UTC offset (Egypt's summer time included). Pure astronomy, no network: «المغرب امتى؟»,
                  "will I make Friday prayer?", iftar in Ramadan.
  eta_to          "How long to Maadi from here?" without touching the trip — the place resolved near the car
                  (Google, as the other place tools) and Mapbox driving-traffic from where the car is now.

Everything numeric here is rounded for the ear by the model (the prompt's SPOKEN NUMBERS rule); results carry
minutes and clock times, never decimals the TTS would mangle.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("routemind.copilot.skills")

# ── Prayer times (pure — tests/test_copilot_skills.py) ────────────────────────
# Egyptian General Authority of Survey: Fajr at 19.5° below the horizon, Isha at 17.5°, Asr when an object's
# shadow equals its length plus its noon shadow (the Shafi'i factor 1), Maghrib at sunset (0.833° for refraction
# and the sun's radius), Dhuhr a minute after the sun crosses the meridian.
FAJR_ANGLE = 19.5
ISHA_ANGLE = 17.5
SUNSET_ANGLE = 0.833
ASR_FACTOR = 1.0
DHUHR_OFFSET_MIN = 1.0
PRAYERS = ("fajr", "sunrise", "dhuhr", "asr", "maghrib", "isha")
PRAYER_AR = {"fajr": "الفجر", "sunrise": "الشروق", "dhuhr": "الضهر", "asr": "العصر", "maghrib": "المغرب",
             "isha": "العشا", "jumuah": "صلاة الجمعة"}


def _julian(d: date) -> float:
    y, m = d.year, d.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d.day + b - 1524.5


def _sun(jd: float) -> Tuple[float, float]:
    """(declination in degrees, equation of time in hours) — the U.S. Naval Observatory approximation."""
    d = jd - 2451545.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    lam = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.degrees(math.atan2(math.cos(e) * math.sin(lam), math.cos(lam))) / 15.0
    decl = math.degrees(math.asin(math.sin(e) * math.sin(lam)))
    eqt = q / 15.0 - (ra % 24)
    eqt = (eqt + 12) % 24 - 12
    return decl, eqt


def _hour_angle(angle: float, lat: float, decl: float) -> Optional[float]:
    la, de = math.radians(lat), math.radians(decl)
    c = (-math.sin(math.radians(angle)) - math.sin(la) * math.sin(de)) / (math.cos(la) * math.cos(de))
    if not -1.0 <= c <= 1.0:
        return None
    return math.degrees(math.acos(c)) / 15.0


def prayer_times(lat: float, lng: float, day: date, utc_offset_min: int) -> Dict[str, float]:
    """Local clock hours (e.g. 17.73) for the six times of `day` at (lat, lng)."""
    decl, eqt = _sun(_julian(day) + 0.5 - lng / 360.0)
    noon = 12.0 + utc_offset_min / 60.0 - lng / 15.0 - eqt
    out: Dict[str, float] = {"dhuhr": noon + DHUHR_OFFSET_MIN / 60.0}
    for name, ang, sign in (("fajr", FAJR_ANGLE, -1), ("sunrise", SUNSET_ANGLE, -1),
                            ("maghrib", SUNSET_ANGLE, 1), ("isha", ISHA_ANGLE, 1)):
        h = _hour_angle(ang, lat, decl)
        if h is not None:
            out[name] = noon + sign * h
    # Asr: the sun's altitude where shadow = factor + tan(|lat − decl|)
    alt = math.degrees(math.atan(1.0 / (ASR_FACTOR + math.tan(math.radians(abs(lat - decl))))))
    h = _hour_angle(-alt, lat, decl)
    if h is not None:
        out["asr"] = noon + h
    return out


def clock(h: float) -> str:
    """17.733 → "17:44" (rounded to the minute, as the calendars print)."""
    m = int(round((h % 24) * 60)) % (24 * 60)
    return f"{m // 60}:{m % 60:02d}"


def _local_now(ctx: Dict[str, Any], now_utc: Optional[datetime] = None) -> Tuple[datetime, int]:
    try:
        off = int(ctx.get("utc_offset_min"))
    except (TypeError, ValueError):
        off = 120                                   # Cairo standard time when the phone did not say
    now = (now_utc or datetime.now(timezone.utc)) + timedelta(minutes=off)
    return now.replace(tzinfo=None), off


def prayers_answer(ctx: Dict[str, Any], which: Optional[str] = None,
                   now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
    if lat is None or lng is None:
        return {"available": False, "note": "No position in hand — say you can't tell the prayer times "
                                            "without the location right now."}
    now, off = _local_now(ctx, now_utc)
    today = prayer_times(float(lat), float(lng), now.date(), off)
    now_h = now.hour + now.minute / 60.0
    upcoming = [(p, today[p]) for p in PRAYERS if p in today and p != "sunrise" and today[p] > now_h]
    if upcoming:
        nxt, nxt_h, nxt_day = upcoming[0][0], upcoming[0][1], "today"
    else:
        tomorrow = prayer_times(float(lat), float(lng), now.date() + timedelta(days=1), off)
        nxt, nxt_h, nxt_day = "fajr", tomorrow["fajr"] + 24.0, "tomorrow"
    friday = now.weekday() == 4
    if friday and nxt == "dhuhr":
        nxt_label = "jumuah"
    else:
        nxt_label = nxt
    out: Dict[str, Any] = {
        "available": True,
        "method": "Egyptian General Authority of Survey",
        "local_time_now": clock(now_h),
        "today": {p: clock(today[p]) for p in PRAYERS if p in today},
        "next": {"prayer": nxt_label, "arabic": PRAYER_AR[nxt_label], "at": clock(nxt_h), "day": nxt_day,
                 "in_min": int(round((nxt_h - now_h) * 60))},
    }
    if which in today:
        out["asked"] = {"prayer": which, "arabic": PRAYER_AR[which], "at": clock(today[which]),
                        "passed": today[which] <= now_h}
    eta = ctx.get("eta_min", ctx.get("remaining_time_min"))
    try:
        if eta is not None:
            arrive_h = now_h + float(eta) / 60.0
            out["arrival_local"] = clock(arrive_h)
            out["arrives_before_next_prayer"] = arrive_h <= nxt_h
    except (TypeError, ValueError):
        pass
    out["note"] = ("Say the one time they asked about (or the next prayer and how long until it), as a clock "
                   "time the Egyptian way; if they are driving to make it, say whether the ETA arrives before it. "
                   "Never list all five unless asked.")
    return out


# ── Departure planning on the prediction model ────────────────────────────────
PLAN_STEP_MIN = 15
PLAN_MAX_HOURS = 12
PLAN_DEFAULT_HOURS = 3
FREE_FLOW_KMH = 45.0                 # when the trip data carries no free-flow time for the stretch


def _route_pts(ctx: Dict[str, Any]) -> List[Tuple[float, float]]:
    pts = []
    for p in (ctx.get("route") or []):
        try:
            pts.append((float(p[1]), float(p[0])))          # [[lng, lat]] → (lat, lng)
        except (TypeError, ValueError, IndexError):
            continue
    return pts


def _length_km(pts: List[Tuple[float, float]]) -> float:
    tot = 0.0
    for a, b in zip(pts, pts[1:]):
        dlat, dlng = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
        x = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlng / 2) ** 2
        tot += 2 * 6371.0 * math.asin(math.sqrt(x))
    return tot


def _sample(pts: List[Tuple[float, float]], n: int = 60) -> List[Dict[str, float]]:
    if len(pts) <= n:
        return [{"lat": a, "lng": b} for a, b in pts]
    step = (len(pts) - 1) / (n - 1)
    return [{"lat": pts[int(round(i * step))][0], "lng": pts[int(round(i * step))][1]} for i in range(n)]


def slot_starts(now: datetime, hours: float, day: str) -> List[datetime]:
    """Departure candidates: the next quarter-hours for `hours` (today), or 06:00–23:00 tomorrow."""
    if day == "tomorrow":
        d = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
        return [d + timedelta(minutes=PLAN_STEP_MIN * i) for i in range(int(17 * 60 / PLAN_STEP_MIN) + 1)]
    first = now.replace(second=0, microsecond=0)
    first += timedelta(minutes=(-first.minute) % PLAN_STEP_MIN)
    n = int(max(0.5, min(PLAN_MAX_HOURS, hours)) * 60 / PLAN_STEP_MIN) + 1
    return [first + timedelta(minutes=PLAN_STEP_MIN * i) for i in range(n)]


def summarize_plan(slots: List[Tuple[datetime, float]]) -> Dict[str, Any]:
    """slots: (departure, predicted minutes). The best window, now, the worst — pure."""
    if not slots:
        return {}
    best = min(slots, key=lambda s: s[1])
    worst = max(slots, key=lambda s: s[1])
    first = slots[0]
    # the best WINDOW: every slot within 2 minutes of the best, contiguous around it
    i = slots.index(best)
    lo = i
    while lo > 0 and slots[lo - 1][1] <= best[1] + 2:
        lo -= 1
    hi = i
    while hi < len(slots) - 1 and slots[hi + 1][1] <= best[1] + 2:
        hi += 1
    return {
        "first_slot": {"leave": first[0].strftime("%H:%M"), "minutes": int(round(first[1]))},
        "best": {"leave": best[0].strftime("%H:%M"), "minutes": int(round(best[1])),
                 "arrive": (best[0] + timedelta(minutes=best[1])).strftime("%H:%M")},
        "best_window": {"from": slots[lo][0].strftime("%H:%M"), "to": slots[hi][0].strftime("%H:%M")},
        "worst": {"leave": worst[0].strftime("%H:%M"), "minutes": int(round(worst[1]))},
        "saving_vs_first_min": int(round(first[1] - best[1])),
        "spread_min": int(round(worst[1] - best[1])),
    }


async def plan_departure(ctx: Dict[str, Any], trip: str = "this_route", hours: float = PLAN_DEFAULT_HOURS,
                         day: str = "today", now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    pts = _route_pts(ctx)
    if len(pts) < 2:
        return {"available": False, "note": "No route in hand to plan — say the departure planner needs the "
                                            "trip on the map (a destination)."}
    if trip == "return_trip":
        pts = list(reversed(pts))
    try:
        from model.tier1 import predict_route_duration
        from services.junction_mapper import (COVERAGE_MIN_PCT, calculate_route_coverage,
                                              map_route_to_junctions)
    except Exception as e:                                         # pragma: no cover — the model's imports
        logger.error(f"plan_departure imports failed: {e}")
        return {"available": False, "note": "The prediction model isn't available right now — say so; offer "
                                            "the usual Cairo rhythm only if they ask."}
    way = _sample(pts)
    try:
        coverage = calculate_route_coverage(way)
    except Exception as e:
        logger.warning(f"plan_departure coverage failed: {e}")
        return {"available": False, "note": "The prediction model isn't available right now — say so."}
    if coverage < COVERAGE_MIN_PCT:
        return {"available": False, "coverage": round(coverage, 2),
                "note": "RouteMind's prediction covers Cairo and Giza roads; this route is mostly outside them. "
                        "Say plainly you can't predict this road by hour; offer to check live traffic instead."}
    km = _length_km(pts)
    base_s = int(km / FREE_FLOW_KMH * 3600)
    if trip == "this_route":
        try:                                   # the trip's own typical time beats a flat speed when we have it
            eta, dl = ctx.get("eta_min"), ctx.get("traffic_delay_min") or 0
            if eta is not None:
                base_s = max(60, int((float(eta) - max(0.0, float(dl))) * 60 / 1.15))
        except (TypeError, ValueError):
            pass
    now, _ = _local_now(ctx, now_utc)
    starts = slot_starts(now, hours, day)
    try:
        junctions = [j["junction_id"] for j in map_route_to_junctions(way)]

        def run() -> List[Tuple[datetime, float]]:
            return [(t, predict_route_duration(junctions, t, base_s)["predicted_seconds"] / 60.0) for t in starts]
        slots = await asyncio.get_running_loop().run_in_executor(None, run)
    except Exception as e:
        logger.warning(f"plan_departure prediction failed: {e}")
        return {"available": False, "note": "The prediction didn't run — say the planner isn't available now."}
    out = {"available": True, "trip": trip, "day": day, "route_km": int(round(km)), "coverage": round(coverage, 2)}
    out.update(summarize_plan(slots))
    out["note"] = ("These are RouteMind's PREDICTED drive times by departure time for this road (learned per "
                   "junction, by weekday and hour) — not live traffic. Give the best time to leave (or the best "
                   "window) and what it saves versus leaving at the first slot, in one or two sentences, rounded "
                   "for the ear. If the spread is under 5 minutes, say it barely matters when they leave.")
    return out


# ── How long to a place from here (no trip change) ────────────────────────────
async def eta_to(ctx: Dict[str, Any], query: str) -> Dict[str, Any]:
    from api import copilot as base
    from api.places import _mapbox_duration
    lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
    q = (query or "").strip()
    if lat is None or lng is None or not q:
        return {"found": False, "note": "No position or no place — ask which place they mean."}
    place = await base._resolve_place(ctx, q)
    if not place:
        return {"found": False, "requested": q, "note": "Nothing matched near here — say so and ask for a "
                                                       "clearer name or area."}
    try:
        secs = await _mapbox_duration(f"{lng},{lat};{place['lng']},{place['lat']}", base._client)
    except Exception as e:
        logger.warning(f"eta_to directions failed: {e}")
        secs = None
    if secs is None:
        return {"found": True, "requested": q, "name": place.get("name"),
                "note": "Found it but the drive time didn't come back — say so briefly."}
    now, _ = _local_now(ctx)
    arrive = now + timedelta(seconds=secs)
    return {"found": True, "requested": q, "name": place.get("name"),
            "minutes_from_here": int(round(secs / 60.0)), "arrive_local": arrive.strftime("%H:%M"),
            "distance_km": round((place.get("distance_m") or 0) / 1000.0),
            "note": "Drive time from where the car is now, with live traffic. Say the minutes (and the arrival "
                    "time if useful). The trip is NOT changed — offer change_destination or add_stop only if they "
                    "sound like they want to go."}
