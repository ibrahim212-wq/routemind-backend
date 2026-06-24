"""
api/places.py
POST /api/places/nearby

Server-side proxy to Google Places API (New) so the app can find nearby places
(for "add a stop" in navigation) WITHOUT shipping the Google API key. The same
endpoint is reused later by the AI assistant's "add nearest gas station" action.

Reuses the existing server-side GOOGLE_MAPS_API_KEY (same env var as
services/google_traffic.py) and the direct-httpx pattern from api/assistant.py.

The key is read from env only, used solely in the X-Goog-Api-Key header, and is
NEVER returned to the client or logged.
"""

import os
import math
import logging
from typing import Optional, List

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("routemind.places")
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────

GOOGLE_KEY        = os.getenv("GOOGLE_MAPS_API_KEY", "")
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_TEXT_URL   = "https://places.googleapis.com/v1/places:searchText"

# Lean field mask → only what we return (keeps cost + response size down).
PLACES_FIELD_MASK = "places.displayName,places.location,places.formattedAddress"

DEFAULT_RADIUS_M = 5000.0   # nearby search circle around the user
MAX_LIMIT        = 10

# Category → Google Places (New) primary type. Kept 1:1 and explicit so the app's
# allowed categories can't silently map to an unexpected Google type.
_CATEGORY_TO_TYPE = {
    "gas_station": "gas_station",
    "pharmacy":    "pharmacy",
    "atm":         "atm",
    "restaurant":  "restaurant",
    "cafe":        "cafe",
}

# ── Models ──────────────────────────────────────────────────────────────────--

class NearbyRequest(BaseModel):
    lat:      float
    lng:      float
    category: Optional[str] = None   # gas_station|pharmacy|atm|restaurant|cafe
    query:    Optional[str] = None   # free-text place name (AI / future search)
    limit:    int           = 1

class PlaceResult(BaseModel):
    name:       str
    lat:        float
    lng:        float
    address:    Optional[str] = None
    distance_m: int

class NearbyResponse(BaseModel):
    results: List[PlaceResult] = []

# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance in metres."""
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_places(data: dict, origin_lat: float, origin_lng: float) -> List[PlaceResult]:
    """Normalize a Google Places (New) response → distance-sorted PlaceResult list."""
    out: List[PlaceResult] = []
    for p in (data.get("places") or []):
        loc = p.get("location") or {}
        lat = loc.get("latitude")
        lng = loc.get("longitude")
        if lat is None or lng is None:
            continue
        name = ((p.get("displayName") or {}).get("text") or "").strip()
        if not name:
            continue
        out.append(PlaceResult(
            name=name,
            lat=float(lat),
            lng=float(lng),
            address=(p.get("formattedAddress") or None),
            distance_m=int(round(_haversine_m(origin_lat, origin_lng, float(lat), float(lng)))),
        ))
    out.sort(key=lambda r: r.distance_m)   # nearest first
    return out

# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/places/nearby", response_model=NearbyResponse)
async def places_nearby(body: NearbyRequest):
    """
    Find nearby places by category (stop picker) or free-text query (AI / search).
    Always returns HTTP 200 with { "results": [...] } — empty on any failure.
    """
    if not GOOGLE_KEY:
        logger.error("GOOGLE_MAPS_API_KEY env var is not set")
        return NearbyResponse(results=[])

    limit = max(1, min(MAX_LIMIT, body.limit or 1))

    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   GOOGLE_KEY,          # server-side only — never sent to client
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }

    try:
        if body.query and body.query.strip():
            # ── Free-text search (places:searchText) biased around the user ──────
            payload = {
                "textQuery": body.query.strip(),
                "maxResultCount": limit,
                "locationBias": {
                    "circle": {
                        "center": {"latitude": body.lat, "longitude": body.lng},
                        "radius": DEFAULT_RADIUS_M,
                    }
                },
            }
            url = PLACES_TEXT_URL
        else:
            # ── Category nearby search (places:searchNearby), nearest first ──────
            gtype = _CATEGORY_TO_TYPE.get((body.category or "").lower())
            if gtype is None:
                logger.warning(f"places_nearby: missing/unknown category={body.category!r}")
                return NearbyResponse(results=[])
            payload = {
                "includedTypes": [gtype],
                "maxResultCount": limit,
                "rankPreference": "DISTANCE",
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": body.lat, "longitude": body.lng},
                        "radius": DEFAULT_RADIUS_M,
                    }
                },
            }
            url = PLACES_NEARBY_URL

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

    except httpx.TimeoutException:
        logger.warning("Google Places request timed out")
        return NearbyResponse(results=[])
    except httpx.HTTPStatusError as e:
        # Log status + body for debugging, but never the key.
        logger.error(f"Google Places HTTP {e.response.status_code}: {e.response.text[:200]}")
        return NearbyResponse(results=[])
    except Exception as e:
        logger.error(f"Unexpected places error: {e}")
        return NearbyResponse(results=[])

    results = _parse_places(data, body.lat, body.lng)[:limit]
    logger.info(f"places_nearby: mode={'text' if body.query else 'category'} "
                f"category={body.category} hits={len(results)}")
    return NearbyResponse(results=results)
