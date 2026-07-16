"""
services/google_traffic.py

Google Maps Routes API — historical traffic predictions للقاهرة.

Fixes:
  - UTC offset: Cairo = UTC+3, departure times تتحول لـ UTC قبل إرسالها
  - now comparison: نقارن Cairo time بـ Cairo time
  - via waypoints: intermediate waypoints كـ passthrough مش mandatory stops
"""

import os
import asyncio
import httpx
import logging
from datetime import datetime, date, timezone, timedelta

logger = logging.getLogger("routemind.google_traffic")

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

CAIRO_UTC_OFFSET = 3  # Cairo = UTC+3

# ─────────────────────────────────────────────────────────────────
# Smart time distribution
# ─────────────────────────────────────────────────────────────────

WEEKDAY_TIMES = [
    (5, 0), (6, 0),
    (7, 0), (7, 30), (8, 0), (8, 30), (9, 0), (9, 30),
    (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 0), (17, 0), (17, 30), (18, 0), (19, 0), (21, 0),
]

FRIDAY_TIMES = [
    (6, 0), (8, 0), (9, 0), (10, 0), (11, 0),
    (12, 0), (12, 30), (13, 0), (13, 30), (14, 0),
    (15, 0), (16, 0), (17, 0), (17, 30), (18, 0),
    (19, 0), (20, 0), (21, 0), (22, 0), (23, 0),
]

SATURDAY_TIMES = [
    (6, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
    (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (17, 30),
    (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0),
    (23, 30), (23, 45),
]


def get_smart_times(target_date: date) -> list[tuple[int, int]]:
    dow = target_date.weekday()
    if dow == 4:
        return FRIDAY_TIMES
    elif dow == 5:
        return SATURDAY_TIMES
    else:
        return WEEKDAY_TIMES


def cairo_now() -> datetime:
    """الوقت الحالي بتوقيت القاهرة (naive datetime)."""
    return datetime.utcnow() + timedelta(hours=CAIRO_UTC_OFFSET)


def cairo_to_utc_iso(cairo_dt: datetime) -> str:
    """تحويل Cairo naive datetime لـ UTC ISO string لـ Google API."""
    utc_dt = cairo_dt - timedelta(hours=CAIRO_UTC_OFFSET)
    return utc_dt.replace(tzinfo=timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────
# Waypoint sampling
# ─────────────────────────────────────────────────────────────────

def _sample_waypoints(waypoints: list[dict], max_count: int = 8) -> list[dict]:
    """
    بياخد sample من الـ waypoints عشان نبعتهم لـ Google كـ via points.
    بنشيل الأول والأخير (هم origin/dest) ونأخد موزعين من الباقي.
    """
    if not waypoints or len(waypoints) <= 2:
        return []

    middle = waypoints[1:-1]
    if not middle:
        return []

    if len(middle) <= max_count:
        return middle

    step = len(middle) / max_count
    return [middle[int(i * step)] for i in range(max_count)]


# ─────────────────────────────────────────────────────────────────
# Google Routes API helpers
# ─────────────────────────────────────────────────────────────────

def _make_headers() -> dict:
    return {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   GOOGLE_KEY,
        "X-Goog-FieldMask": "routes.duration",
    }


def _make_body(
    origin_lat:   float,
    origin_lng:   float,
    dest_lat:     float,
    dest_lng:     float,
    departure_cairo_dt: datetime | None = None,
    traffic_aware: bool = True,
    waypoints: list[dict] | None = None,
) -> dict:
    body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude":  origin_lat,
                    "longitude": origin_lng,
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude":  dest_lat,
                    "longitude": dest_lng,
                }
            }
        },
        "travelMode":        "DRIVE",
        "routingPreference": "TRAFFIC_AWARE_OPTIMAL" if traffic_aware else "TRAFFIC_UNAWARE",
    }

    # Intermediate waypoints كـ via (passthrough) — مش mandatory stops
    if waypoints:
        body["intermediates"] = [
            {
                "via": True,
                "location": {
                    "latLng": {
                        "latitude":  wp["lat"],
                        "longitude": wp["lng"],
                    }
                },
            }
            for wp in waypoints
        ]

    # departure time: نحول Cairo → UTC
    if departure_cairo_dt and traffic_aware:
        body["departureTime"] = cairo_to_utc_iso(departure_cairo_dt)

    return body


def _parse_duration(resp_json: dict) -> int | None:
    routes = resp_json.get("routes", [])
    if not routes:
        return None
    dur_str = routes[0].get("duration", "")
    if not dur_str:
        return None
    return int(dur_str.rstrip("s"))


# ─────────────────────────────────────────────────────────────────
# Free flow baseline (TRAFFIC_UNAWARE)
# ─────────────────────────────────────────────────────────────────

async def get_google_free_flow(
    origin_lat: float, origin_lng: float,
    dest_lat:   float, dest_lng:   float,
    waypoints: list[dict] | None = None,
) -> int | None:
    """
    يجيب الـ theoretical free flow duration بدون traffic.
    مش بيستخدم departure time.
    """
    sampled = _sample_waypoints(waypoints or [])
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                ROUTES_URL,
                json=_make_body(
                    origin_lat, origin_lng,
                    dest_lat,   dest_lng,
                    traffic_aware=False,
                    waypoints=sampled,
                ),
                headers=_make_headers(),
            )
        if resp.status_code != 200:
            logger.warning(f"Google free flow: {resp.status_code}")
            return None
        dur = _parse_duration(resp.json())
        logger.info(f"Google free flow (UNAWARE): {dur}s")
        return dur
    except Exception as e:
        logger.error(f"Google free flow error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Single departure time request
# ─────────────────────────────────────────────────────────────────

async def _get_google_duration(
    origin_lat: float, origin_lng: float,
    dest_lat:   float, dest_lng:   float,
    departure_cairo_dt: datetime,
    client: httpx.AsyncClient,
    waypoints: list[dict] | None = None,
) -> int | None:
    try:
        resp = await client.post(
            ROUTES_URL,
            json=_make_body(
                origin_lat, origin_lng,
                dest_lat,   dest_lng,
                departure_cairo_dt=departure_cairo_dt,
                traffic_aware=True,
                waypoints=waypoints,
            ),
            headers=_make_headers(),
            timeout=12.0,
        )
        if resp.status_code != 200:
            logger.warning(
                f"Google Routes {resp.status_code} "
                f"@ {departure_cairo_dt.strftime('%H:%M')} Cairo"
            )
            return None
        return _parse_duration(resp.json())
    except Exception as e:
        logger.error(
            f"Google Routes error @ {departure_cairo_dt.strftime('%H:%M')}: {e}"
        )
        return None


# ─────────────────────────────────────────────────────────────────
# Stream Google plan
# ─────────────────────────────────────────────────────────────────

async def stream_google_plan(
    origin_lat: float, origin_lng: float,
    dest_lat:   float, dest_lng:   float,
    target_date: date,
    waypoints: list[dict] | None = None,
):
    """
    Async generator — بيبعت (hour, minute, duration_seconds).

    - departure times بتوقيت القاهرة وبتتحول لـ UTC قبل إرسالها لـ Google
    - لو الوقت في الماضي → نضيف 7 أيام (نفس اليوم الأسبوع الجاي)
    - intermediate waypoints كـ via (passthrough)
    """
    times   = get_smart_times(target_date)
    sampled = _sample_waypoints(waypoints or [])
    sem     = asyncio.Semaphore(3)
    queue: asyncio.Queue = asyncio.Queue()
    results: dict = {}
    lock    = asyncio.Lock()

    # الوقت الحالي بتوقيت القاهرة
    now_cairo = cairo_now()

    async def fetch_one(h: int, m: int):
        # Cairo naive datetime
        dep_cairo = datetime(
            target_date.year, target_date.month, target_date.day, h, m
        )

        # لو الوقت في الماضي → نفس اليوم الأسبوع الجاي
        if dep_cairo <= now_cairo:
            dep_cairo += timedelta(days=7)

        async with sem:
            async with httpx.AsyncClient() as client:
                dur = await _get_google_duration(
                    origin_lat, origin_lng,
                    dest_lat,   dest_lng,
                    dep_cairo, client,
                    waypoints=sampled,
                )

        if dur:
            async with lock:
                results[(h, m)] = dur
            await queue.put((h, m, dur))

    tasks    = [asyncio.create_task(fetch_one(h, m)) for h, m in times]
    received = 0
    total    = len(times)

    while received < total:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=15.0)
            yield item
            received += 1
        except asyncio.TimeoutError:
            if all(t.done() for t in tasks):
                break

    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(
        f"Google plan: {len(results)}/{total} slots | "
        f"date={target_date} ({target_date.strftime('%A')}) | "
        f"via_waypoints={len(sampled)}"
    )


# ─────────────────────────────────────────────────────────────────
# Route-level live/predicted congestion (for the intelligent scanner)
# ─────────────────────────────────────────────────────────────────

# True-free-flow baseline cache. Free flow is time-invariant on the scale of a
# day, so one Routes call per route per 12h is enough (scans repeat every 10min).
# Key: rounded endpoints + waypoint count. Value: (free_seconds, fetched_ts).
_FREE_FLOW_TTL = 12 * 3600
_free_flow_cache: dict = {}


def _next_night_utc() -> datetime:
    """Next 3:30 AM Cairo, as UTC — Google's predicted duration then is true
    free flow for the route (same source & units as the live call)."""
    now_cairo = datetime.utcnow() + timedelta(hours=CAIRO_UTC_OFFSET)
    night = now_cairo.replace(hour=3, minute=30, second=0, microsecond=0)
    if night <= now_cairo:
        night += timedelta(days=1)
    return (night - timedelta(hours=CAIRO_UTC_OFFSET)).replace(tzinfo=timezone.utc)


async def get_google_route_congestion(
    origin_lat: float, origin_lng: float,
    dest_lat:   float, dest_lng:   float,
    free_flow_seconds: float | None = None,
    departure_utc: datetime | None = None,
    waypoints: list[dict] | None = None,
) -> dict | None:
    """
    Route-level congestion for the intelligent scanner.

        jam = clamp((live / free_flow - 1) / 0.50, 0, 1)     (slot-aligned)

    Two hard-won calibration facts (measured Cairo, Thursday rush 2026-07-16):

    1. BASELINE — Google's TRAFFIC_UNAWARE duration is a *typical-traffic*
       duration, not free flow. In Cairo "typical" already includes heavy
       traffic, so live/unaware compressed every rush hour to ratio≈1.0-1.3 and
       the scanner under-reported delays (+8 min shown when the true delay vs
       free flow was +15-20). The baseline is now Google's PREDICTED duration at
       the next 3:30 AM Cairo — true free flow from the same source — cached per
       route for 12h. TRAFFIC_UNAWARE and the caller's stored duration remain
       fallbacks only.

    2. ROUTE PINNING — with origin/dest only, TRAFFIC_AWARE_OPTIMAL happily
       routes AROUND the congestion the user will actually sit in. `waypoints`
       (sampled from the trip's stored route_polyline) pin both the live and the
       baseline request to the user's planned corridor.

    Returns {live_seconds, free_flow_seconds, congestion_ratio, jam_factor,
             delay_seconds, baseline_source} or None on failure.
    """
    now = datetime.now(timezone.utc)
    if departure_utc is None or departure_utc <= now + timedelta(minutes=2):
        departure_utc = now + timedelta(minutes=2)
    dep_iso = (departure_utc.astimezone(timezone.utc)
               .replace(microsecond=0).isoformat().replace("+00:00", "Z"))

    # Callers pass intermediate points only (no origin/dest), so sample inline —
    # _sample_waypoints would strip the first/last as endpoints and lose two.
    sampled: list[dict] = []
    if waypoints:
        if len(waypoints) <= 8:
            sampled = list(waypoints)
        else:
            step = len(waypoints) / 8.0
            sampled = [waypoints[int(i * step)] for i in range(8)]

    def _body(pref: str, dep: str | None) -> dict:
        b = {
            "origin":      {"location": {"latLng": {"latitude": origin_lat, "longitude": origin_lng}}},
            "destination": {"location": {"latLng": {"latitude": dest_lat,   "longitude": dest_lng}}},
            "travelMode":  "DRIVE",
            "routingPreference": pref,
        }
        if sampled:
            b["intermediates"] = [
                {"via": True,
                 "location": {"latLng": {"latitude": wp["lat"], "longitude": wp["lng"]}}}
                for wp in sampled
            ]
        if dep:
            b["departureTime"] = dep
        return b

    cache_key = (round(origin_lat, 4), round(origin_lng, 4),
                 round(dest_lat, 4),   round(dest_lng, 4), len(sampled))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            live_resp = await client.post(
                ROUTES_URL, json=_body("TRAFFIC_AWARE_OPTIMAL", dep_iso),
                headers=_make_headers())
            if live_resp.status_code != 200:
                logger.warning(f"Google congestion: live {live_resp.status_code} {live_resp.text[:120]}")
                return None
            live = _parse_duration(live_resp.json())
            if not live:
                logger.warning("Google congestion: no live route returned")
                return None

            # ── True free-flow baseline (cached) ──
            free = None
            baseline_source = "night_predicted"
            import time as _time
            hit = _free_flow_cache.get(cache_key)
            if hit and (_time.time() - hit[1]) < _FREE_FLOW_TTL:
                free = hit[0]
                baseline_source = "night_predicted_cached"
            else:
                night_iso = (_next_night_utc().replace(microsecond=0)
                             .isoformat().replace("+00:00", "Z"))
                night_resp = await client.post(
                    ROUTES_URL, json=_body("TRAFFIC_AWARE", night_iso),
                    headers=_make_headers())
                if night_resp.status_code == 200:
                    free = _parse_duration(night_resp.json())
                if free:
                    _free_flow_cache[cache_key] = (free, _time.time())

            if not free:
                unaware_resp = await client.post(
                    ROUTES_URL, json=_body("TRAFFIC_UNAWARE", None),
                    headers=_make_headers())
                free = (_parse_duration(unaware_resp.json())
                        if unaware_resp.status_code == 200 else None)
                baseline_source = "traffic_unaware"
            if free is None and free_flow_seconds and free_flow_seconds > 0:
                free = int(free_flow_seconds)
                baseline_source = "stored_fallback"
    except Exception as e:
        logger.error(f"Google congestion error: {e}")
        return None

    if not free or free <= 0:
        logger.warning("Google congestion: no free-flow baseline")
        return None

    # A predicted night duration can very occasionally exceed a quiet live one
    # (route variants); never report negative congestion.
    ratio = max(1.0, live / free)
    jam   = max(0.0, min(1.0, (ratio - 1.0) / 0.50))
    delay = max(0, live - free)
    logger.info(
        f"Google congestion: live={live}s free={free}s ({baseline_source}) "
        f"ratio={ratio:.2f} jam={jam:.2f} delay={delay//60}min via={len(sampled)}wp"
    )
    return {
        "live_seconds":      live,
        "free_flow_seconds": free,
        "congestion_ratio":  round(ratio, 3),
        "jam_factor":        round(jam, 3),   # 0..1, same scale as plan-drive slots
        "delay_seconds":     delay,
        "baseline_source":   baseline_source,
    }