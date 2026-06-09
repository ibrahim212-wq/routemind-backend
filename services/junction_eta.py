"""
services/junction_eta.py

Per-junction ETA calculation for RouteMind Intelligent Scan.
لكل junction على المسار، نحسب متى المتوقع إن العربية تكون عنده.
"""

from __future__ import annotations
import math
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def haversine_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(max(0, min(1, a))))


def _cumulative_distances(polyline: List[Tuple[float, float]]) -> List[float]:
    if not polyline:
        return []
    cum = [0.0]
    for i in range(1, len(polyline)):
        cum.append(cum[-1] + haversine_km(polyline[i-1], polyline[i]))
    return cum


def compute_junction_etas(
    route_polyline: List[Tuple[float, float]],
    junctions_on_route: List[Dict[str, Any]],
    total_duration_seconds: float,
    departure_time: datetime,
) -> List[Dict[str, Any]]:
    """
    لكل junction، نحسب الـ ETA من وقت انطلاق الرحلة.

    Args:
        route_polyline:        [(lat, lon), ...] من Mapbox geometry
        junctions_on_route:    output بتاع map_route_to_junctions() مباشرة
                               {junction_id, junction_idx, latitude, longitude, ...}
        total_duration_seconds: baseSecs من Mapbox
        departure_time:         وقت بداية الرحلة

    Returns:
        list of dicts مرتبة بـ ETA تصاعدياً:
        {junction_id, junction_idx, latitude, longitude,
         distance_from_origin_km, eta_seconds, eta_datetime}
    """
    if not route_polyline or len(route_polyline) < 2:
        logger.warning("compute_junction_etas: polyline فاضي أو قصير")
        return []

    if total_duration_seconds <= 0:
        logger.warning(f"compute_junction_etas: duration غير صالح ({total_duration_seconds})")
        return []

    if not junctions_on_route:
        return []

    cumulative    = _cumulative_distances(route_polyline)
    total_dist_km = cumulative[-1]

    if total_dist_km <= 0:
        logger.warning("compute_junction_etas: total distance = 0")
        return []

    avg_speed = (total_dist_km / total_duration_seconds) * 3600
    logger.info(f"Route: {total_dist_km:.1f}km | {total_duration_seconds/60:.0f}min | {avg_speed:.0f}km/h")

    results = []
    for j in junctions_on_route:
        # keys بتاعت junction_mapper
        jlat = j.get("latitude")
        jlon = j.get("longitude")
        jid  = j.get("junction_id")

        if jlat is None or jlon is None or jid is None:
            continue

        # أقرب نقطة على الـ polyline
        target = (jlat, jlon)
        min_d, min_i = float("inf"), 0
        for i, p in enumerate(route_polyline):
            d = haversine_km(target, p)
            if d < min_d:
                min_d, min_i = d, i

        dist_from_origin = cumulative[min_i]
        eta_sec = (dist_from_origin / total_dist_km) * total_duration_seconds

        results.append({
            "junction_id":           jid,
            "junction_idx":          j.get("junction_idx"),
            "latitude":              jlat,
            "longitude":             jlon,
            "distance_from_origin_km": round(dist_from_origin, 3),
            "eta_seconds":           round(eta_sec, 1),
            "eta_datetime":          departure_time + timedelta(seconds=eta_sec),
            "distance_to_route_km":  round(min_d, 3),
        })

    results.sort(key=lambda x: x["eta_seconds"])
    logger.info(f"ETA حُسب لـ {len(results)} junction")
    return results