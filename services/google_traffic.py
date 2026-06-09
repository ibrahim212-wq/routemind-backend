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