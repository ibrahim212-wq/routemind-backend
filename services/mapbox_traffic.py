"""
services/mapbox_traffic.py

Mapbox Directions API — live traffic data للقاهرة.
أدق بكتير من TomTom لأن Mapbox عنده probe data حقيقية من مصر.
"""

import os
import asyncio
import httpx
import logging

logger = logging.getLogger("routemind.mapbox_traffic")

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
_DIRECTIONS  = "https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords}"


async def _get_duration(
    origin_lat: float, origin_lng: float,
    dest_lat:   float, dest_lng:   float,
    profile: str,
    client: httpx.AsyncClient,
) -> int | None:
    coords = f"{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
    try:
        r = await client.get(
            _DIRECTIONS.format(profile=profile, coords=coords),
            params={"access_token": MAPBOX_TOKEN, "overview": "false"},
            timeout=10.0,
        )
        if r.status_code != 200:
            logger.warning(f"Mapbox {profile} → {r.status_code}")
            return None
        routes = r.json().get("routes", [])
        return int(routes[0]["duration"]) if routes else None
    except Exception as e:
        logger.error(f"Mapbox {profile} error: {e}")
        return None


async def get_mapbox_congestion(
    origin_lat: float, origin_lng: float,
    dest_lat:   float, dest_lng:   float,
) -> dict | None:
    """
    يجيب live congestion للـ plan drive.
    بيعمل 2 requests concurrent:
      1. driving-traffic → live duration
      2. driving         → free flow duration

    Returns:
        {live_seconds, free_flow_seconds, congestion_ratio}
    أو None لو فشل.
    """
    async with httpx.AsyncClient() as client:
        live, free = await asyncio.gather(
            _get_duration(origin_lat, origin_lng, dest_lat, dest_lng,
                          "driving-traffic", client),
            _get_duration(origin_lat, origin_lng, dest_lat, dest_lng,
                          "driving", client),
        )

    if not live or not free or free == 0:
        logger.warning("Mapbox congestion: failed to get durations")
        return None

    ratio = live / free
    logger.info(
        f"Mapbox congestion: live={live}s free={free}s ratio={ratio:.2f}"
    )
    return {
        "live_seconds":      live,
        "free_flow_seconds": free,
        "congestion_ratio":  round(ratio, 3),
    }


async def get_mapbox_segment_speed(
    from_lat: float, from_lng: float,
    to_lat:   float, to_lng:   float,
) -> dict | None:
    """
    يجيب live speed على segment للـ intelligent scan.
    بيستخدم Mapbox speed annotations.

    Returns:
        {jam_factor, current_speed, free_flow_speed,
         congestion_ratio, speed_reduction, confidence}
    أو None لو فشل.
    """
    coords = f"{from_lng},{from_lat};{to_lng},{to_lat}"

    async with httpx.AsyncClient(timeout=8.0) as client:
        traffic_r, free_r = await asyncio.gather(
            client.get(
                _DIRECTIONS.format(profile="driving-traffic", coords=coords),
                params={"access_token": MAPBOX_TOKEN,
                        "overview": "false", "annotations": "speed"},
            ),
            client.get(
                _DIRECTIONS.format(profile="driving", coords=coords),
                params={"access_token": MAPBOX_TOKEN,
                        "overview": "false", "annotations": "speed"},
            ),
        )

    try:
        if traffic_r.status_code != 200:
            return None

        def _avg(resp) -> float | None:
            routes = resp.json().get("routes", [])
            if not routes or not routes[0].get("legs"):
                return None
            speeds = routes[0]["legs"][0].get("annotation", {}).get("speed", [])
            return sum(speeds) / len(speeds) if speeds else None

        cur  = _avg(traffic_r)
        free = _avg(free_r) if free_r.status_code == 200 else None

        if cur is None:
            return None
        if free is None or free <= 0:
            free = cur

        ratio = max(0.0, 1.0 - cur / free)
        jam   = round(ratio * 10, 2)

        return {
            "jam_factor":       jam,
            "current_speed":    round(cur  * 3.6, 1),       # km/h
            "free_flow_speed":  round(free * 3.6, 1),       # km/h
            "congestion_ratio": round(ratio, 3),
            "speed_reduction":  round((free - cur) * 3.6, 1),
            "confidence":       0.85,
        }
    except Exception as e:
        logger.error(f"Mapbox segment error: {e}")
        return None