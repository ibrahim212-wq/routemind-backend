"""
services/tomtom.py

Two responsibilities:
  1. Live junction readings  → used by scan_route during monitoring
  2. Route duration planning → used by plan_drive when junctions < 5
                               (routes on Ring Road / outside training data)
"""

import os
import asyncio
import time
import httpx
from datetime import datetime, date
import logging

logger = logging.getLogger("routemind.tomtom")

TOMTOM_KEY       = os.getenv("TOMTOM_API_KEY", "")
FLOW_URL         = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
ROUTE_URL        = "https://api.tomtom.com/routing/1/calculateRoute/{coords}/json"

# ── Plan cache ────────────────────────────────────────────────────────────────
_plan_cache: dict = {}
CACHE_TTL_SECONDS = 3600  # ساعة

# ── 27 strategic departure times ─────────────────────────────────────────────
# Dense at morning/evening peak, hourly elsewhere
PLAN_SLOTS: list[tuple[int, int]] = [
    (5,  0), (6,  0),                                          # early morning
    (7,  0), (7, 30),                                          # pre-peak
    (8,  0), (8, 15), (8, 30), (8, 45),                       # morning peak ×4
    (9,  0), (9, 30),                                          # post-peak
    (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),    # daytime
    (16, 0), (16, 30),                                         # pre-peak
    (17, 0), (17,15), (17,30), (17,45),                        # evening peak ×4
    (18, 0), (18,30),                                          # post-peak
    (19, 0), (20, 0), (21, 0),                                 # night
]  # total = 27


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LIVE JUNCTION READINGS  (used by scan_route)
# ══════════════════════════════════════════════════════════════════════════════

async def get_junction_reading(
    junction_id: str,
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> dict | None:
    """
    Current traffic reading for one junction.
    Returns dict or None on failure.
    """
    try:
        resp = await client.get(
            FLOW_URL,
            params={"point": f"{lat},{lon}", "key": TOMTOM_KEY},
            timeout=5.0,
        )
        if resp.status_code != 200:
            logger.warning(f"TomTom flow {resp.status_code} for {junction_id}")
            return None

        fd = resp.json().get("flowSegmentData", {})
        current_speed    = float(fd.get("currentSpeed",   30.0))
        free_flow_speed  = float(fd.get("freeFlowSpeed",  30.0))
        confidence       = float(fd.get("confidence",      0.5))

        congestion_ratio = (
            current_speed / free_flow_speed if free_flow_speed > 0 else 1.0
        )
        jam_factor = max(0.0, (1.0 - congestion_ratio) * 10.0)

        return {
            "junction_id":      junction_id,
            "timestamp":        datetime.utcnow().isoformat(),
            "current_speed":    current_speed,
            "free_flow_speed":  free_flow_speed,
            "jam_factor":       round(jam_factor, 3),
            "congestion_ratio": round(congestion_ratio, 3),
            "speed_reduction":  round(max(0.0, free_flow_speed - current_speed), 3),
            "confidence":       round(confidence, 3),
            "delay_seconds":    0.0,
        }

    except Exception as e:
        logger.error(f"TomTom flow error for {junction_id}: {e}")
        return None


async def get_readings_for_junctions(
    junctions: list[dict],
) -> dict[str, dict]:
    """
    Parallel live readings for a list of junctions.
    Returns {junction_id: reading_dict}
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            get_junction_reading(
                j["junction_id"], j["latitude"], j["longitude"], client
            )
            for j in junctions
        ]
        readings = await asyncio.gather(*tasks)

    return {
        j["junction_id"]: r
        for j, r in zip(junctions, readings)
        if r is not None
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ROUTE DURATION PLANNING  (used by plan_drive fallback)
# ══════════════════════════════════════════════════════════════════════════════

async def _get_route_duration(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    depart_dt: datetime,
    client: httpx.AsyncClient,
) -> int | None:
    """
    TomTom predicted travel time (seconds) for one departure datetime.
    Returns None on any failure so callers can skip gracefully.
    """
    coords     = f"{origin_lat},{origin_lng}:{dest_lat},{dest_lng}"
    depart_str = depart_dt.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        resp = await client.get(
            ROUTE_URL.format(coords=coords),
            params={
                "key":        TOMTOM_KEY,
                "departAt":   depart_str,
                "traffic":    "true",
                "travelMode": "car",
            },
            timeout=12.0,
        )
        if resp.status_code != 200:
            logger.warning(f"TomTom route {resp.status_code} at {depart_str}")
            return None

        routes = resp.json().get("routes", [])
        if not routes:
            return None

        return int(routes[0]["summary"]["travelTimeInSeconds"])

    except Exception as e:
        logger.error(f"TomTom route error at {depart_str}: {e}")
        return None


async def get_tomtom_plan(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    target_date: date,
) -> list[tuple[int, int, int]] | None:
    """
    Makes 27 TomTom route calls at strategic departure times.
    Uses Semaphore(5) to avoid 429 rate limiting.
    Caches results for 1 hour per route + day of week.

    Returns:
        List of (hour, minute, duration_seconds) for successful calls,
        or None if every call failed (total API failure).
    """
    # Cache key: route + day of week
    cache_key = (
        round(origin_lat, 3), round(origin_lng, 3),
        round(dest_lat, 3),   round(dest_lng, 3),
        target_date.weekday(),
    )

    # Check cache
    now = time.time()
    if cache_key in _plan_cache:
        results, ts = _plan_cache[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            logger.info("TomTom plan: cache hit")
            return results

    # Semaphore: max 5 concurrent requests to avoid 429
    sem = asyncio.Semaphore(5)

    async def _limited(h, m):
        async with sem:
            async with httpx.AsyncClient() as client:
                return await _get_route_duration(
                    origin_lat, origin_lng,
                    dest_lat,   dest_lng,
                    datetime(target_date.year, target_date.month, target_date.day, h, m),
                    client,
                )

    durations = await asyncio.gather(*[_limited(h, m) for h, m in PLAN_SLOTS])

    results = [
        (h, m, dur)
        for (h, m), dur in zip(PLAN_SLOTS, durations)
        if dur is not None
    ]

    if not results:
        logger.error("TomTom plan: all 27 calls failed")
        return None

    # Save to cache
    _plan_cache[cache_key] = (results, now)

    logger.info(f"TomTom plan: {len(results)}/27 calls succeeded")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STREAMING PLAN  (used by plan_drive_stream)
# ══════════════════════════════════════════════════════════════════════════════

async def _get_route_duration_with_retry(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    depart_dt: datetime,
    client: httpx.AsyncClient,
    retries: int = 3,
) -> int | None:
    """نفس _get_route_duration بس مع retry للـ 429."""
    for attempt in range(retries):
        coords     = f"{origin_lat},{origin_lng}:{dest_lat},{dest_lng}"
        depart_str = depart_dt.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            resp = await client.get(
                ROUTE_URL.format(coords=coords),
                params={
                    "key":        TOMTOM_KEY,
                    "departAt":   depart_str,
                    "traffic":    "true",
                    "travelMode": "car",
                },
                timeout=12.0,
            )
            if resp.status_code == 429:
                await asyncio.sleep(1.5 ** attempt)
                continue
            if resp.status_code != 200:
                return None
            routes = resp.json().get("routes", [])
            if not routes:
                return None
            return int(routes[0]["summary"]["travelTimeInSeconds"])
        except Exception as e:
            logger.error(f"TomTom route error at {depart_str}: {e}")
            return None
    return None


async def stream_tomtom_plan(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    target_date: date,
):
    """
    Async generator — يعمل yield لكل TomTom result لما ييجي.
    بيرتب الـ slots بالأقرب للوقت الحالي عشان الـ user يشوف أقرب وقت أول.
    بيستخدم asyncio.as_completed عشان يبعت كل result فوراً.
    """
    # رتب الـ slots بالأقرب للوقت الحالي
    now_minutes = datetime.now().hour * 60 + datetime.now().minute

    def _dist_from_now(h: int, m: int) -> int:
        diff = h * 60 + m - now_minutes
        return diff if diff >= 0 else diff + 24 * 60

    sorted_slots = sorted(PLAN_SLOTS, key=lambda hm: _dist_from_now(*hm))

    sem = asyncio.Semaphore(2)

    async def _one(h, m):
        async with sem:
            async with httpx.AsyncClient() as client:
                dur = await _get_route_duration_with_retry(
                    origin_lat, origin_lng,
                    dest_lat,   dest_lng,
                    datetime(target_date.year, target_date.month, target_date.day, h, m),
                    client,
                )
                return (h, m, dur)

    tasks = [asyncio.ensure_future(_one(h, m)) for h, m in sorted_slots]

    for future in asyncio.as_completed(tasks):
        h, m, dur = await future
        if dur is not None:
            yield h, m, dur