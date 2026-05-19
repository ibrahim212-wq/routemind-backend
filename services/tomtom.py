"""
services/tomtom.py
Gets live traffic readings from TomTom for specific junctions.
Called by scan_route during monitoring windows.
"""

import os
import asyncio
import httpx
from datetime import datetime
import logging

logger = logging.getLogger("routemind.tomtom")

TOMTOM_KEY = os.getenv("TOMTOM_API_KEY", "")
BASE_URL   = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


async def get_junction_reading(
    junction_id: str,
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> dict | None:
    """
    Gets current traffic reading for one junction from TomTom.
    Returns dict with traffic fields or None on failure.
    """
    try:
        resp = await client.get(
            BASE_URL,
            params={"point": f"{lat},{lon}", "key": TOMTOM_KEY},
            timeout=5.0,
        )
        if resp.status_code != 200:
            logger.warning(f"TomTom {resp.status_code} for {junction_id}")
            return None

        fd = resp.json().get("flowSegmentData", {})
        current_speed   = float(fd.get("currentSpeed",       30.0))
        free_flow_speed = float(fd.get("freeFlowSpeed",      30.0))
        confidence      = float(fd.get("confidence",          0.5))

        speed_reduction = max(0.0, free_flow_speed - current_speed)
        congestion_ratio = (
            current_speed / free_flow_speed if free_flow_speed > 0 else 1.0
        )
        # Approximate jam_factor on 0-10 scale from congestion_ratio
        jam_factor = max(0.0, (1.0 - congestion_ratio) * 10.0)
        delay_seconds = 0.0  # would need distance for accurate calc

        return {
            "junction_id":     junction_id,
            "timestamp":       datetime.utcnow().isoformat(),
            "current_speed":   current_speed,
            "free_flow_speed": free_flow_speed,
            "jam_factor":      round(jam_factor, 3),
            "congestion_ratio": round(congestion_ratio, 3),
            "speed_reduction": round(speed_reduction, 3),
            "confidence":      round(confidence, 3),
            "delay_seconds":   delay_seconds,
        }

    except Exception as e:
        logger.error(f"TomTom error for {junction_id}: {e}")
        return None


async def get_readings_for_junctions(
    junctions: list[dict],
) -> dict[str, dict]:
    """
    Gets current readings for a list of junctions in parallel.

    Args:
        junctions: List of {junction_id, latitude, longitude}

    Returns:
        Dict mapping junction_id → reading dict
    """
    results = {}
    async with httpx.AsyncClient() as client:
        tasks = [
            get_junction_reading(
                j["junction_id"], j["latitude"], j["longitude"], client
            )
            for j in junctions
        ]
        readings = await asyncio.gather(*tasks)

    for j, reading in zip(junctions, readings):
        if reading is not None:
            results[j["junction_id"]] = reading

    return results