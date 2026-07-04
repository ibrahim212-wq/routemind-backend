"""
services/overpass.py

Minimal, isolated OpenStreetMap Overpass fallback for the along-route POI search.

This is NOT the primary source — api/places.py uses Google Places first. Overpass
is queried ONLY when Google returns zero results (or errors) for a category, so
the user still gets something in areas where Google's Egypt POI coverage is thin.

Kept deliberately small: one bbox query per category, a short retry, and a tiny
in-memory TTL cache keyed by (category, rounded bbox). No new dependency — async
httpx (already used across the backend) and a plain dict cache (same ad-hoc
in-memory caching style as services/mapbox_traffic.py; the stack has no Redis).

Returns raw POI dicts with (lat, lng); the caller applies the corridor geometry
filter, so this module knows nothing about the route.
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("routemind.overpass")

# Reliable public endpoint; overridable so it can be tuned / mirrored later.
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OVERPASS_TIMEOUT_S = int(os.getenv("OVERPASS_TIMEOUT_S", "25"))
OVERPASS_CACHE_TTL_S = int(os.getenv("OVERPASS_CACHE_TTL_S", "300"))

# overpass-api.de returns 406 for requests without a User-Agent — always send one.
_HEADERS = {"User-Agent": "RouteMind/1.0 (routemind backend; along-route POI fallback)"}

# category → list of Overpass QL filter clauses (applied to BOTH node and way).
_CATEGORY_CLAUSES: Dict[str, List[str]] = {
    "fuel":       ['["amenity"="fuel"]'],
    "restaurant": ['["amenity"="restaurant"]', '["amenity"="fast_food"]'],
    "cafe":       ['["amenity"="cafe"]'],
    "atm":        ['["amenity"="atm"]', '["amenity"="bank"]["atm"="yes"]'],
    "parking":    ['["amenity"="parking"]'],
}

# Generic label when a POI has neither name:ar, name, nor brand.
_GENERIC_LABEL: Dict[str, str] = {
    "fuel":       "Fuel station",
    "restaurant": "Restaurant",
    "cafe":       "Café",
    "atm":        "ATM",
    "parking":    "Parking",
}

# (category, rounded_bbox) → (timestamp, results)
_cache: Dict[Tuple[str, Tuple[float, float, float, float]], Tuple[float, List[dict]]] = {}


def _build_query(category: str, bbox: Tuple[float, float, float, float]) -> str:
    """Build an Overpass QL query for nodes AND ways (center) within bbox."""
    south, west, north, east = bbox
    b = f"({south:.5f},{west:.5f},{north:.5f},{east:.5f})"
    parts: List[str] = []
    for clause in _CATEGORY_CLAUSES.get(category, []):
        parts.append(f"  node{clause}{b};")
        parts.append(f"  way{clause}{b};")
    body = "\n".join(parts)
    return f"[out:json][timeout:{OVERPASS_TIMEOUT_S}];\n(\n{body}\n);\nout center tags;"


def _pick_name(tags: dict, category: str) -> str:
    """Prefer name:ar, then name, then brand, else a generic per-category label."""
    return (tags.get("name:ar") or tags.get("name") or tags.get("brand")
            or _GENERIC_LABEL.get(category, "Place"))


def _element_to_poi(el: dict, category: str) -> Optional[dict]:
    tags = el.get("tags") or {}
    if el.get("type") == "node":
        lat, lng = el.get("lat"), el.get("lon")
    else:  # way / relation → use the computed center
        center = el.get("center") or {}
        lat, lng = center.get("lat"), center.get("lon")
    if lat is None or lng is None:
        return None

    house = tags.get("addr:housenumber")
    street = tags.get("addr:street")
    address = " ".join(x for x in (house, street) if x) or None

    return {
        "id": f"osm/{el.get('type')}/{el.get('id')}",
        "name": _pick_name(tags, category),
        "lat": float(lat),
        "lng": float(lng),
        "brand": tags.get("brand"),
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "opening_hours": tags.get("opening_hours"),
        "address": address,
    }


async def fetch_pois(category: str, bbox: Tuple[float, float, float, float]
                     ) -> List[dict]:
    """
    Query Overpass for `category` within `bbox` (south, west, north, east).
    Returns a list of POI dicts (may be empty). NEVER raises — any failure logs
    and returns []. One short retry on timeout. Results are cached per
    (category, rounded bbox) for OVERPASS_CACHE_TTL_S seconds.
    """
    if category not in _CATEGORY_CLAUSES:
        return []

    key = (category, tuple(round(v, 3) for v in bbox))  # ~100 m bbox rounding
    hit = _cache.get(key)
    now = time.time()
    if hit and (now - hit[0]) < OVERPASS_CACHE_TTL_S:
        return hit[1]

    query = _build_query(category, bbox)
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=OVERPASS_TIMEOUT_S + 5) as client:
                resp = await client.post(OVERPASS_URL, data={"data": query}, headers=_HEADERS)
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                pois = [p for p in (_element_to_poi(e, category) for e in elements) if p]
                _cache[key] = (now, pois)
                logger.info(f"Overpass fallback: category={category} hits={len(pois)}")
                return pois
        except httpx.TimeoutException:
            logger.warning(f"Overpass timeout (attempt {attempt}) for category={category}")
            if attempt == 1:
                await asyncio.sleep(1.0)
            continue
        except Exception as e:
            logger.error(f"Overpass error for category={category}: {e}")
            return []
    return []
