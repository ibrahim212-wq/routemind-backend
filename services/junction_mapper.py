"""
services/junction_mapper.py
Maps lat/lng coordinates to nearest junction IDs using the metadata.
"""

import math
import pandas as pd
from model.loader import ModelLoader

# Max distance to consider a junction "on" the route
# 2.5km is tight enough to exclude Ring Road junctions
# but loose enough for inner-city routes
COVERAGE_MAX_KM  = 2.5

# Min % of waypoints that must have a nearby junction
# for us to trust our model
COVERAGE_MIN_PCT = 0.70


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0, min(1, a))))


def find_nearest_junction(
    lat: float,
    lon: float,
    max_km: float = 10.0,
) -> dict | None:
    """
    Finds the nearest junction within max_km.
    Returns junction dict or None if nothing is close enough.
    """
    meta = ModelLoader.get_meta()
    best_dist = float("inf")
    best_row  = None

    for _, row in meta.iterrows():
        d = haversine_km(lat, lon, float(row["latitude"]), float(row["longitude"]))
        if d < best_dist:
            best_dist = d
            best_row  = row

    if best_dist > max_km or best_row is None:
        return None

    return {
        "junction_id":  best_row["junction_id"],
        "junction_idx": int(best_row["junction_idx"]),
        "latitude":     float(best_row["latitude"]),
        "longitude":    float(best_row["longitude"]),
        "distance_km":  round(best_dist, 3),
    }


def map_route_to_junctions(waypoints: list[dict]) -> list[dict]:
    """
    Maps route waypoints to unique junction IDs in route order.
    Uses max_km=10.0 to always find the nearest junction.
    """
    seen   = set()
    result = []

    for wp in waypoints:
        j = find_nearest_junction(wp["lat"], wp["lng"], max_km=10.0)
        if j and j["junction_id"] not in seen:
            seen.add(j["junction_id"])
            result.append(j)

    return result


def calculate_route_coverage(waypoints: list[dict]) -> float:
    """
    Returns 0.0–1.0: the fraction of waypoints that have a training-data
    junction within COVERAGE_MAX_KM (2.5 km).

    Interpretation:
      ≥ 0.70  →  our model has good data for this route  → use model
      < 0.70  →  large parts of the route are outside    → use TomTom
                 our training data (e.g. Ring Road)

    This is more reliable than counting junctions because it directly
    answers "do we have data for THIS route?" — not just "are there
    junctions somewhere nearby?".
    """
    if not waypoints:
        return 0.0

    covered = sum(
        1 for wp in waypoints
        if find_nearest_junction(wp["lat"], wp["lng"], max_km=COVERAGE_MAX_KM) is not None
    )
    coverage = covered / len(waypoints)
    return round(coverage, 3)


def get_junction_by_id(junction_id: str) -> dict | None:
    """Returns junction metadata by ID."""
    meta = ModelLoader.get_meta()
    row  = meta[meta["junction_id"] == junction_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "junction_id":  junction_id,
        "junction_idx": int(r["junction_idx"]),
        "latitude":     float(r["latitude"]),
        "longitude":    float(r["longitude"]),
    }