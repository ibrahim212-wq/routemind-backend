"""
services/junction_mapper.py
Maps lat/lng coordinates to nearest junction IDs using the metadata.
"""

import math
import pandas as pd
from model.loader import ModelLoader


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, min(1, a))))


def find_nearest_junction(lat: float, lon: float, max_km: float = 2.0) -> dict | None:
    """
    Finds the nearest junction to given coordinates within max_km.
    Returns junction dict or None if too far.
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
    Maps a list of {lat, lng} waypoints to junction IDs.
    Removes duplicate junctions and junctions too far from waypoints.

    Args:
        waypoints: List of {lat: float, lng: float}

    Returns:
        List of unique junction dicts in route order
    """
    seen = set()
    result = []

    for wp in waypoints:
        j = find_nearest_junction(wp["lat"], wp["lng"])
        if j and j["junction_id"] not in seen:
            seen.add(j["junction_id"])
            result.append(j)

    return result


def get_junction_by_id(junction_id: str) -> dict | None:
    """Returns junction metadata by ID."""
    meta = ModelLoader.get_meta()
    row = meta[meta["junction_id"] == junction_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "junction_id": junction_id,
        "junction_idx": int(r["junction_idx"]),
        "latitude": float(r["latitude"]),
        "longitude": float(r["longitude"]),
    }