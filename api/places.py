"""
api/places.py
Google Places proxy + "Add stop along the route" backend.

All endpoints proxy Google (and, for POIs, an OSM/Overpass fallback) server-side
so the app never ships the Google API key. Endpoints:

  POST /api/places/nearby            — nearby by category / free-text (existing)
  POST /api/along-route/pois         — category POIs along the REMAINING route
  POST /api/along-route/detours      — extra driving time to visit each candidate
  GET  /api/search/autocomplete      — Google Places Autocomplete (any place)
  GET  /api/search/place/{place_id}  — Google Place Details for a chosen result

Reuses the existing server-side GOOGLE_MAPS_API_KEY (same env var as
services/google_traffic.py) and the direct-httpx pattern from api/assistant.py.
Detours reuse the Mapbox Directions pattern + MAPBOX_TOKEN from
services/mapbox_traffic.py. Corridor geometry lives in services/route_geometry.py;
the Overpass fallback lives in services/overpass.py.

Every key is read from env only, used solely in request headers/params, and is
NEVER returned to the client or logged.

── Example curls ──────────────────────────────────────────────────────────────
  # 1) POIs along a route (Tahrir → New Cairo), category = fuel
  curl -s localhost:8080/api/along-route/pois -H 'Content-Type: application/json' -d '{
    "category":"fuel","max_results":10,
    "current_location":{"lat":30.0444,"lng":31.2357},
    "route":[[31.2357,30.0444],[31.3260,30.0500],[31.4370,30.0270]]}'

  # 2) Detour time to two candidates
  curl -s localhost:8080/api/along-route/detours -H 'Content-Type: application/json' -d '{
    "current_location":{"lat":30.0444,"lng":31.2357},
    "destination":{"lat":30.0270,"lng":31.4370},
    "candidates":[{"id":"a","lat":30.0500,"lng":31.3260}]}'

  # 3) Autocomplete   4) Place details
  curl -s 'localhost:8080/api/search/autocomplete?q=carrefour&lat=30.0444&lng=31.2357'
  curl -s 'localhost:8080/api/search/place/PLACE_ID_HERE'
"""

import os
import math
import asyncio
import logging
from typing import Optional, List

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from services import route_geometry as geo
from services import overpass

logger = logging.getLogger("routemind.places")
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────

GOOGLE_KEY        = os.getenv("GOOGLE_MAPS_API_KEY", "")
MAPBOX_TOKEN      = os.getenv("MAPBOX_TOKEN", "")   # same var as mapbox_traffic.py
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_TEXT_URL   = "https://places.googleapis.com/v1/places:searchText"
PLACES_AUTOCOMPLETE_URL = "https://places.googleapis.com/v1/places:autocomplete"
PLACES_DETAILS_URL      = "https://places.googleapis.com/v1/places/"   # + place_id
MAPBOX_DIRECTIONS_URL   = "https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords}"

# Lean field mask → only what we return (keeps cost + response size down).
PLACES_FIELD_MASK = "places.displayName,places.location,places.formattedAddress"
# Along-route needs the place id too (so the client can fetch details / detour).
PLACES_POI_FIELD_MASK = "places.id,places.displayName,places.location,places.formattedAddress"

DEFAULT_RADIUS_M = 5000.0   # nearby search circle around the user
MAX_LIMIT        = 10

# ── Along-route tunables (env-overridable so they can be tuned in Cloud Run) ────
CORRIDOR_BUFFER_M    = float(os.getenv("CORRIDOR_BUFFER_M", "1500"))  # Overpass bbox pad
CORRIDOR_WIDTH_M     = float(os.getenv("CORRIDOR_WIDTH_M", "800"))    # perp filter + circle radius
MAX_DETOUR_CANDIDATES = int(os.getenv("MAX_DETOUR_CANDIDATES", "8"))
MAX_CORRIDOR_ANCHORS  = int(os.getenv("MAX_CORRIDOR_ANCHORS", "12"))  # Google circle-search centres
# Two-band corridor anchoring (fixes "nearest on route" misses): the old single
# pass let sample_along WIDEN the spacing on long routes (12 anchors on a 70 km
# route → ~6.4 km between 800 m circles → most of the corridor never searched,
# so a station 4 km ahead was missed and a 70 km one "won"). Near band uses a
# spacing smaller than 2× the circle radius → CONTIGUOUS coverage of the first
# ~18 km; a sparse far band still finds distant options.
CORRIDOR_NEAR_SPACING_M = float(os.getenv("CORRIDOR_NEAR_SPACING_M", "1500"))
CORRIDOR_NEAR_ANCHORS   = int(os.getenv("CORRIDOR_NEAR_ANCHORS", "12"))
CORRIDOR_FAR_ANCHORS    = int(os.getenv("CORRIDOR_FAR_ANCHORS", "8"))


def corridor_anchors(remaining, cum):
    """Anchor centres for the corridor circle-search: gapless near band (first
    NEAR_SPACING×NEAR_ANCHORS metres) + sparse far band over the remainder."""
    near_len = CORRIDOR_NEAR_SPACING_M * CORRIDOR_NEAR_ANCHORS
    if not cum or cum[-1] <= near_len:
        return geo.sample_along(remaining, cum, CORRIDOR_NEAR_SPACING_M,
                                CORRIDOR_NEAR_ANCHORS)
    split = next(k for k in range(len(cum)) if cum[k] >= near_len)
    prefix, pcum = remaining[:split + 1], cum[:split + 1]
    rest = remaining[split:]
    rcum = [c - cum[split] for c in cum[split:]]
    anchors = geo.sample_along(prefix, pcum, CORRIDOR_NEAR_SPACING_M,
                               CORRIDOR_NEAR_ANCHORS)
    anchors += geo.sample_along(rest, rcum, CORRIDOR_NEAR_SPACING_M,
                                CORRIDOR_FAR_ANCHORS)
    return anchors

# Category → Google Places (New) primary type. Kept 1:1 and explicit so the app's
# allowed categories can't silently map to an unexpected Google type.
_CATEGORY_TO_TYPE = {
    "gas_station": "gas_station",
    "pharmacy":    "pharmacy",
    "atm":         "atm",
    "restaurant":  "restaurant",
    "cafe":        "cafe",
}

# Along-route category vocabulary (the app's 5 stop categories) → Google type.
# Independent of the /places/nearby vocabulary above by design.
_ALONG_ROUTE_TYPE = {
    "fuel":       "gas_station",
    "restaurant": "restaurant",
    "cafe":       "cafe",
    "atm":        "atm",
    "parking":    "parking",
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

def _fix_mojibake(s: Optional[str], debug_label: Optional[str] = None) -> Optional[str]:
    """
    Repair "double UTF-8 encoding" mojibake (e.g. 'محطة غاز' arriving as
    mangled control-character-laden text) seen in Google Places (New) text
    fields for some Egyptian POI records — confirmed on BOTH displayName and
    formattedAddress, per record (a record's name and address are corrupted
    independently; it is NOT "address always, name never" — apply this to
    every Google text field you display). Pattern: the original UTF-8 bytes
    were, at some point in that data's history, decoded with a single-byte
    codec and then correctly UTF-8-encoded again for transmission — reversing
    it means recovering those original bytes, then decoding as UTF-8.

    The codec for that recovery step MUST be latin-1, not cp1252, even though
    cp1252 is the far more common real-world mojibake codec in general. Proof:
    a huge share of Arabic letters (e.g. م U+0645) UTF-8-encode to a byte in
    the 0x80–0x9F range as their continuation byte (م → 0xD9 0x85). latin-1 is
    a full 256-value identity mapping — byte 0x85 decodes to literal control
    codepoint U+0085 — which is exactly what a byte-for-byte "wrong single-byte
    codec" misread does. cp1252 does NOT map 0x80-0x9F to identity; it maps
    that range to typographic symbols instead (0x85 → U+2026 ellipsis, not
    U+0085) and several of those byte values are simply undefined in cp1252.
    So if the upstream corruption actually happened via latin-1 (confirmed by
    finding literal U+0085 in real corrupted data), trying to REVERSE it with
    .encode('cp1252') raises UnicodeEncodeError for any string containing that
    control codepoint — silently failing to repair on exactly the most common
    Arabic letters. latin-1's encode step never raises for codepoints 0–255,
    so it can always attempt the reversal.

    Safe by construction: correctly-decoded non-Latin text (e.g. real Arabic)
    contains codepoints ≥ U+0100, outside latin-1's 0–255 range, so the
    .encode('latin-1') step raises and this returns the string UNCHANGED. The
    subsequent .decode('utf-8') is the second safety net — a string that's
    entirely latin-1-representable but whose bytes aren't valid UTF-8 (e.g. a
    plain accented-Latin address that was never corrupted) also passes through
    unchanged. Only strings that are BOTH entirely latin-1-representable AND
    whose bytes form valid UTF-8 get "repaired" — exactly the mojibake
    fingerprint.

    `debug_label`, when given, logs every call: a DEBUG line when no repair
    was needed/possible, a WARNING line with the before/after values when a
    repair actually fired. Temporary diagnostic aid — safe to leave in (only
    fires when debug_label is passed, and repairs are rare/cheap to log).
    """
    if not s:
        return s
    try:
        repaired = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        if debug_label:
            logger.debug(f"[MOJIBAKE DEBUG] {debug_label}: no repair "
                         f"(not cp1252/utf-8 mojibake — {e}) input={s!r}")
        return s
    if repaired == s:
        # Pure-ASCII strings round-trip through cp1252->utf-8 unchanged (ASCII is
        # a subset of both) — that's not a "repair", just a no-op; don't log it
        # as one or every plain English field would look like it got fixed.
        if debug_label:
            logger.debug(f"[MOJIBAKE DEBUG] {debug_label}: round-trip no-op "
                         f"(already clean) value={s!r}")
        return s
    if debug_label:
        logger.warning(f"[MOJIBAKE DEBUG] {debug_label}: REPAIRED "
                       f"before={s!r} after={repaired!r}")
    return repaired


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
        name = _fix_mojibake(name, debug_label="places/nearby name")
        out.append(PlaceResult(
            name=name,
            lat=float(lat),
            lng=float(lng),
            address=_fix_mojibake(p.get("formattedAddress") or None,
                                  debug_label="places/nearby address"),
            distance_m=int(round(_haversine_m(origin_lat, origin_lng, float(lat), float(lng)))),
        ))
    out.sort(key=lambda r: r.distance_m)   # nearest first
    return out


async def _google_post(url: str, payload: dict, field_mask: str,
                       client: Optional[httpx.AsyncClient] = None,
                       debug_label: Optional[str] = None) -> Optional[dict]:
    """
    POST to a Google Places (New) endpoint with the server-side key + field mask.
    Returns the parsed JSON dict, or None on ANY failure (no key, HTTP error,
    timeout). Never raises, never logs/returns the key. Reuses a caller-supplied
    AsyncClient when given (so concurrent corridor searches share connections).

    When `debug_label` is given, logs the HTTP status for every call, AND — this
    is the case a plain except-block can't catch — logs the raw response body
    when Google answers 200 OK with an EMPTY result set, since that's not an
    exception at all and was previously silent. Diagnostic aid for "why did the
    corridor search come back empty"; safe to leave on (only fires for corridor
    calls, which pass debug_label; /places/nearby and autocomplete don't).
    """
    if not GOOGLE_KEY:
        logger.error("GOOGLE_MAPS_API_KEY env var is not set")
        return None
    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   GOOGLE_KEY,          # server-side only — never sent to client
        "X-Goog-FieldMask": field_mask,
    }
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=10.0)
    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=10.0)
        if debug_label:
            logger.info(f"[GOOGLE DEBUG] {debug_label}: HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        if debug_label and not (data.get("places") or data.get("suggestions")):
            # 200 OK but nothing found — no exception fires, so log it explicitly.
            logger.warning(f"[GOOGLE DEBUG] {debug_label}: 200 OK, ZERO results. "
                           f"Raw body: {resp.text[:500]!r}")
        return data
    except httpx.TimeoutException:
        logger.warning(f"Google Places request timed out ({debug_label or url})")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"Google Places HTTP {e.response.status_code} "
                     f"({debug_label or url}): {e.response.text[:500]}")
        return None
    except Exception as e:
        logger.error(f"Unexpected places error ({debug_label or url}): {e}")
        return None
    finally:
        if owns:
            await client.aclose()

# ── Shared resolver (used by BOTH the endpoint and the assistant add_stop tool) ─

async def resolve_nearby(
    lat: float,
    lng: float,
    *,
    category: Optional[str] = None,
    query: Optional[str] = None,
    included_type: Optional[str] = None,
    limit: int = 1,
) -> List[PlaceResult]:
    """
    Resolve nearby places via Google Places (New) — category (searchNearby,
    nearest first) or free-text (searchText, biased to the user). Returns a
    distance-sorted list (≤ limit). NEVER raises and NEVER returns/logs the key —
    on any failure (no key, bad category, HTTP error, timeout) returns [].

    included_type (free-text mode only) constrains the text search to one
    Google place type — e.g. query "Master" + included_type "gas_station"
    matches only fuel stations, not any shop whose name contains the word.
    """
    if not GOOGLE_KEY:
        logger.error("GOOGLE_MAPS_API_KEY env var is not set")
        return []

    n = max(1, min(MAX_LIMIT, limit or 1))

    if query and query.strip():
        payload = {
            "textQuery": query.strip(),
            "maxResultCount": n,
            "locationBias": {
                "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": DEFAULT_RADIUS_M}
            },
        }
        if included_type:
            payload["includedType"] = included_type
        url = PLACES_TEXT_URL
    else:
        gtype = _CATEGORY_TO_TYPE.get((category or "").lower())
        if gtype is None:
            logger.warning(f"resolve_nearby: missing/unknown category={category!r}")
            return []
        payload = {
            "includedTypes": [gtype],
            "maxResultCount": n,
            "rankPreference": "DISTANCE",
            "locationRestriction": {
                "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": DEFAULT_RADIUS_M}
            },
        }
        url = PLACES_NEARBY_URL

    data = await _google_post(url, payload, PLACES_FIELD_MASK)
    if not data:
        return []
    return _parse_places(data, lat, lng)[:n]

# ══════════════════════════════════════════════════════════════════════════════
#  Enriched retrieval (the copilot's recommendation layer)
#
#  COST NOTE (approved): the search mask below adds rating/userRatingCount/
#  priceLevel/currentOpeningHours/primaryType → Google's ENTERPRISE SKU
#  ($35 / 1000, vs the $32 Pro the lean mask already bills). reviews +
#  editorialSummary are ENTERPRISE+ATMOSPHERE ($40) and are fetched ONLY in
#  place_details_rich() for a single drilled-into place — never on list search.
# ══════════════════════════════════════════════════════════════════════════════

# Enterprise-tier search mask: identity + quality signals for ranking/compare.
PLACES_RICH_MASK = (
    "places.id,places.displayName,places.location,places.formattedAddress,"
    "places.primaryType,places.rating,places.userRatingCount,places.priceLevel,"
    "places.currentOpeningHours.openNow,places.businessStatus"
)
# Atmosphere-tier details mask (single place, on demand): + reviews + editorial.
PLACE_DETAILS_RICH_MASK = (
    "id,displayName,location,formattedAddress,primaryType,rating,userRatingCount,"
    "priceLevel,currentOpeningHours,regularOpeningHours.weekdayDescriptions,"
    "internationalPhoneNumber,editorialSummary,reviews"
)
_PRICE_WORD = {"PRICE_LEVEL_FREE": "free", "PRICE_LEVEL_INEXPENSIVE": "cheap",
               "PRICE_LEVEL_MODERATE": "moderate", "PRICE_LEVEL_EXPENSIVE": "pricey",
               "PRICE_LEVEL_VERY_EXPENSIVE": "very pricey"}


def _rich_place(p: dict, origin: Optional[tuple] = None) -> Optional[dict]:
    """Normalize one Places (New) result into the copilot's enriched shape.
    origin=(lat,lng) attaches straight-line distance_m when given."""
    loc = p.get("location") or {}
    lat, lng = loc.get("latitude"), loc.get("longitude")
    if lat is None or lng is None:
        return None
    name = _fix_mojibake(((p.get("displayName") or {}).get("text") or "").strip())
    if not name:
        return None
    out = {
        "id": p.get("id"),
        "name": name,
        "lat": float(lat), "lng": float(lng),
        "address": _fix_mojibake(p.get("formattedAddress") or None),
        "primary_type": p.get("primaryType"),
        "rating": p.get("rating"),
        "reviews": p.get("userRatingCount"),
        "price": _PRICE_WORD.get(p.get("priceLevel") or ""),
        "open_now": (p.get("currentOpeningHours") or {}).get("openNow"),
    }
    if origin and lat is not None:
        out["distance_m"] = int(round(_haversine_m(origin[0], origin[1], float(lat), float(lng))))
    return out


async def search_places(
    *,
    lat: float,
    lng: float,
    query: str,
    included_type: Optional[str] = None,
    open_now: bool = False,
    min_rating: Optional[float] = None,
    rank: str = "relevance",              # relevance | distance
    radius_m: float = DEFAULT_RADIUS_M,
    route_polyline: Optional[str] = None,  # encoded polyline5 → search-along-route
    limit: int = 5,
) -> List[dict]:
    """
    Enriched Text Search (New). Free-text semantic query + circle bias around
    (lat,lng) — or, when route_polyline is given, Google's native
    searchAlongRouteParameters (one call, corridor-aware). Returns enriched
    dicts (rating/reviews/price/open_now/primary_type). NEVER raises → [].
    """
    if not GOOGLE_KEY or not query.strip():
        return []
    n = max(1, min(20, limit))
    payload: dict = {"textQuery": query.strip(), "pageSize": n,
                     "rankPreference": "DISTANCE" if rank == "distance" else "RELEVANCE"}
    if included_type:
        payload["includedType"] = included_type
    if open_now:
        payload["openNow"] = True
    if min_rating:
        payload["minRating"] = float(min_rating)
    if route_polyline:
        # Bias to the route corridor; Google ranks by relevance along the path.
        payload["searchAlongRouteParameters"] = {"polyline": {"encodedPolyline": route_polyline}}
        payload["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lng},
                                              "radius": min(radius_m, 50000.0)}}
    else:
        payload["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lng},
                                              "radius": min(radius_m, 50000.0)}}
    data = await _google_post(PLACES_TEXT_URL, payload, PLACES_RICH_MASK,
                              debug_label=f"search_places {query!r}")
    if not data:
        return []
    out = [_rich_place(p, origin=(lat, lng)) for p in (data.get("places") or [])]
    return [p for p in out if p][:n]


async def place_details_rich(place_id: str) -> Optional[dict]:
    """Single-place deep fetch (Atmosphere SKU): reviews + hours + editorial.
    On-demand only. None on failure."""
    if not place_id:
        return None
    data = await _google_get(PLACES_DETAILS_URL + place_id, PLACE_DETAILS_RICH_MASK)
    if not data:
        return None
    base = _rich_place(data) or {}
    ed = (data.get("editorialSummary") or {}).get("text")
    # 3 × 280 chars: enough for one telling quote; a 5×500 payload measurably
    # slowed the model's spoken reply (reviews-latency defect).
    reviews = []
    for r in (data.get("reviews") or [])[:3]:
        txt = _fix_mojibake(((r.get("text") or {}).get("text") or "").strip())
        if txt:
            reviews.append({"rating": r.get("rating"),
                            "when": r.get("relativePublishTimeDescription"),
                            "text": txt[:280]})
    hours = (data.get("regularOpeningHours") or {}).get("weekdayDescriptions")
    base.update({
        "editorial": _fix_mojibake(ed) if ed else None,
        "phone": data.get("internationalPhoneNumber"),
        "hours": hours,
        "review_samples": reviews,
    })
    return base

# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/places/nearby", response_model=NearbyResponse)
async def places_nearby(body: NearbyRequest):
    """
    Find nearby places by category (stop picker) or free-text query (AI / search).
    Always returns HTTP 200 with { "results": [...] } — empty on any failure.
    Thin wrapper over the shared resolve_nearby().
    """
    results = await resolve_nearby(
        body.lat, body.lng, category=body.category, query=body.query, limit=body.limit)
    logger.info(f"places_nearby: mode={'text' if body.query else 'category'} "
                f"category={body.category} hits={len(results)}")
    return NearbyResponse(results=results)


# ══════════════════════════════════════════════════════════════════════════════
#  Add-stop-along-the-route
# ══════════════════════════════════════════════════════════════════════════════

# ── Models ─────────────────────────────────────────────────────────────────────

class LatLng(BaseModel):
    lat: float
    lng: float


class AlongRoutePoisRequest(BaseModel):
    category: str                                   # fuel|restaurant|cafe|atm|parking
    current_location: LatLng
    route: Optional[List[List[float]]] = None       # GeoJSON LineString [[lng,lat],...]
    polyline6: Optional[str] = None                 # alternative: encoded polyline (precision 6)
    max_results: int = 15
    nearest: bool = False                           # True → nearest to the user (ignore route)


class AlongRoutePoi(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    category: str
    distance_from_route_m: int
    along_route_distance_m: int
    brand: Optional[str] = None
    phone: Optional[str] = None
    opening_hours: Optional[str] = None
    address: Optional[str] = None


class AlongRoutePoisResponse(BaseModel):
    source: str                                     # "google" | "overpass" | "none"
    results: List[AlongRoutePoi] = []


class DetourCandidate(BaseModel):
    id: str
    lat: float
    lng: float


class DetoursRequest(BaseModel):
    candidates: List[DetourCandidate]
    current_location: LatLng
    destination: LatLng
    max_candidates: int = MAX_DETOUR_CANDIDATES


class DetourResult(BaseModel):
    id: str
    total_duration_sec: Optional[int] = None
    added_seconds: Optional[int] = None


class DetoursResponse(BaseModel):
    baseline_duration_sec: Optional[int] = None
    results: List[DetourResult] = []


class AutocompleteItem(BaseModel):
    place_id: str
    primary_text: str
    secondary_text: str = ""


class PlaceDetails(BaseModel):
    name: str
    lat: float
    lng: float
    phone: Optional[str] = None
    rating: Optional[float] = None
    address: Optional[str] = None
    opening_hours: Optional[List[str]] = None


# ── Google GET helper (Place Details is a GET; searches above are POST) ─────────

async def _google_get(url: str, field_mask: str, params: Optional[dict] = None
                      ) -> Optional[dict]:
    """GET a Google Places (New) endpoint with key + field mask. None on failure."""
    if not GOOGLE_KEY:
        logger.error("GOOGLE_MAPS_API_KEY env var is not set")
        return None
    headers = {"X-Goog-Api-Key": GOOGLE_KEY, "X-Goog-FieldMask": field_mask}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params or {})
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        logger.warning(f"Google Places GET timed out ({url})")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"Google Places GET HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"Unexpected places GET error: {e}")
        return None


# ── Along-route POIs: Google corridor scan (primary) + Overpass (fallback) ──────

def _parse_route(req: "AlongRoutePoisRequest") -> List:
    """Route input → [(lat,lng), ...] from GeoJSON coords or an encoded polyline6."""
    if req.route:
        return geo.geojson_to_latlng(req.route)
    if req.polyline6:
        return geo.decode_polyline6(req.polyline6)
    return []


async def _google_corridor_pois(gtype: str, anchors: List, n_per_anchor: int = 20
                                ) -> List[dict]:
    """
    Run one Google searchNearby per anchor point CONCURRENTLY (circle radius =
    CORRIDOR_WIDTH_M), merge and dedupe by place id. Returns raw POI dicts
    (id/name/lat/lng/address). Empty on total failure.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        async def _one(i: int, anchor) -> dict:
            payload = {
                "includedTypes": [gtype],
                "maxResultCount": n_per_anchor,
                "rankPreference": "DISTANCE",
                "locationRestriction": {"circle": {
                    "center": {"latitude": anchor[0], "longitude": anchor[1]},
                    "radius": CORRIDOR_WIDTH_M}},
            }
            label = f"corridor anchor {i} ({anchor[0]:.5f},{anchor[1]:.5f}) type={gtype}"
            return await _google_post(PLACES_NEARBY_URL, payload,
                                      PLACES_POI_FIELD_MASK, client=client,
                                      debug_label=label)

        responses = await asyncio.gather(*[_one(i, a) for i, a in enumerate(anchors)])

    seen: set = set()
    pois: List[dict] = []
    for data in responses:
        for p in ((data or {}).get("places") or []):
            loc = p.get("location") or {}
            lat, lng = loc.get("latitude"), loc.get("longitude")
            if lat is None or lng is None:
                continue
            pid = p.get("id") or f"g/{round(float(lat),6)}/{round(float(lng),6)}"
            if pid in seen:
                continue
            seen.add(pid)
            name = ((p.get("displayName") or {}).get("text") or "").strip()
            name = _fix_mojibake(name, debug_label=f"corridor POI {pid} name")
            pois.append({
                "id": pid,
                "name": name or "Place",
                "lat": float(lat),
                "lng": float(lng),
                "brand": None,
                "phone": None,
                "opening_hours": None,
                "address": _fix_mojibake(p.get("formattedAddress") or None,
                                         debug_label=f"corridor POI {pid} address"),
            })
    return pois


def _filter_to_corridor(pois: List[dict], remaining: List, cum: List[float],
                        category: str, max_results: int) -> List[AlongRoutePoi]:
    """
    Keep POIs whose perpendicular distance to the remaining route ≤ CORRIDOR_WIDTH_M,
    attach along-route distance, drop anything behind the user, sort by along-route
    distance ascending, cap to max_results.
    """
    out: List[AlongRoutePoi] = []
    for p in pois:
        perp, along = geo.project_point(remaining, cum, (p["lat"], p["lng"]))
        if perp > CORRIDOR_WIDTH_M or along < 0:      # too far off route / behind user
            continue
        out.append(AlongRoutePoi(
            id=p["id"], name=p["name"], lat=p["lat"], lng=p["lng"], category=category,
            distance_from_route_m=int(round(perp)),
            along_route_distance_m=int(round(along)),
            brand=p.get("brand"), phone=p.get("phone"),
            opening_hours=p.get("opening_hours"), address=p.get("address"),
        ))
    out.sort(key=lambda r: r.along_route_distance_m)
    return out[:max_results]


@router.post("/along-route/pois", response_model=AlongRoutePoisResponse)
async def along_route_pois(req: AlongRoutePoisRequest):
    """
    Category POIs along the REMAINING route (from the vertex nearest the user to
    the destination), within a CORRIDOR_WIDTH_M corridor, nearest-along-route
    first. Google Places is the primary source; if it returns nothing for the
    category we fall back to OSM/Overpass. Always HTTP 200 — empty on any failure.
    """
    category = (req.category or "").lower()
    gtype = _ALONG_ROUTE_TYPE.get(category)
    if gtype is None:
        logger.warning(f"along_route_pois: unknown category={req.category!r}")
        return AlongRoutePoisResponse(source="none", results=[])

    # NEAREST-TO-ME mode: closest of the category to the user's current location,
    # regardless of route. Circle search (searchNearby, rank=DISTANCE) around the
    # user; along_route_distance_m carries the straight-line distance so the
    # client's cards/sort work unchanged.
    if req.nearest:
        cl = req.current_location
        data = await _google_post(PLACES_NEARBY_URL, {
            "includedTypes": [gtype], "maxResultCount": min(req.max_results, 20),
            "rankPreference": "DISTANCE",
            "locationRestriction": {"circle": {
                "center": {"latitude": cl.lat, "longitude": cl.lng},
                "radius": 5000.0}}}, PLACES_POI_FIELD_MASK, debug_label=f"nearest {category}")
        out: List[AlongRoutePoi] = []
        for p in ((data or {}).get("places") or []):
            loc = p.get("location") or {}
            plat, plng = loc.get("latitude"), loc.get("longitude")
            if plat is None or plng is None:
                continue
            d = int(round(_haversine_m(cl.lat, cl.lng, float(plat), float(plng))))
            out.append(AlongRoutePoi(
                id=p.get("id") or f"near/{len(out)}",
                name=_fix_mojibake((p.get("displayName") or {}).get("text") or "") or "Place",
                lat=float(plat), lng=float(plng), category=category,
                distance_from_route_m=d, along_route_distance_m=d,
                address=_fix_mojibake(p.get("formattedAddress") or None)))
        out.sort(key=lambda r: r.along_route_distance_m)
        logger.info(f"along_route_pois NEAREST: category={category} hits={len(out)}")
        return AlongRoutePoisResponse(source="google" if out else "none",
                                      results=out[:req.max_results])

    route = _parse_route(req)
    if len(route) < 2:
        logger.warning("along_route_pois: route needs ≥2 points")
        return AlongRoutePoisResponse(source="none", results=[])

    current = (req.current_location.lat, req.current_location.lng)
    remaining = geo.remaining_route(route, current)
    if len(remaining) < 2:
        return AlongRoutePoisResponse(source="none", results=[])
    cum = geo.cumulative_distances(remaining)
    anchors = corridor_anchors(remaining, cum)   # two-band: gapless near + sparse far

    # 1) Google (primary)
    raw = await _google_corridor_pois(gtype, anchors)
    results = _filter_to_corridor(raw, remaining, cum, category, req.max_results)
    source = "google"
    # Distinguish "Google returned nothing at all" (upstream/API issue) from
    # "Google returned POIs but the corridor filter dropped all of them"
    # (geometry/tuning issue) — otherwise both look identical downstream.
    logger.info(f"[GOOGLE DEBUG] along_route_pois: category={category} gtype={gtype} "
               f"anchors={len(anchors)} google_raw={len(raw)} google_after_filter={len(results)}")

    # 2) Overpass (fallback) — only when Google yielded nothing for this corridor
    if not results:
        bbox = geo.corridor_bbox(remaining, CORRIDOR_BUFFER_M)
        osm_raw = await overpass.fetch_pois(category, bbox)
        results = _filter_to_corridor(osm_raw, remaining, cum, category, req.max_results)
        source = "overpass" if results else "none"

    logger.info(f"along_route_pois: category={category} source={source} "
                f"anchors={len(anchors)} hits={len(results)}")
    return AlongRoutePoisResponse(source=source, results=results)


# ── Detours: extra driving time to visit each candidate (Mapbox Directions) ─────

async def _mapbox_duration(coords: str, client: httpx.AsyncClient) -> Optional[int]:
    """
    driving-traffic duration (sec) for a coord string "lng,lat;lng,lat[;...]".
    Mirrors the Directions call in services/mapbox_traffic.py. None on failure.
    """
    if not MAPBOX_TOKEN:
        logger.error("MAPBOX_TOKEN env var is not set")
        return None
    try:
        r = await client.get(
            MAPBOX_DIRECTIONS_URL.format(profile="driving-traffic", coords=coords),
            params={"access_token": MAPBOX_TOKEN, "overview": "false"},
            timeout=10.0,
        )
        if r.status_code != 200:
            logger.warning(f"Mapbox directions → {r.status_code}")
            return None
        routes = r.json().get("routes", [])
        return int(routes[0]["duration"]) if routes else None
    except Exception as e:
        logger.error(f"Mapbox directions error: {e}")
        return None


@router.post("/along-route/detours", response_model=DetoursResponse)
async def along_route_detours(req: DetoursRequest):
    """
    For each candidate stop, how much extra driving time it adds vs. going
    straight to the destination. Baseline = current→destination; per candidate =
    current→candidate→destination. Runs candidates concurrently; a candidate that
    fails just gets null durations. Always HTTP 200. Uses driving-traffic to match
    the app's remaining-time logic.
    """
    cur = req.current_location
    dst = req.destination
    candidates = req.candidates[: max(1, req.max_candidates or MAX_DETOUR_CANDIDATES)]

    async with httpx.AsyncClient(timeout=10.0) as client:
        baseline_task = _mapbox_duration(f"{cur.lng},{cur.lat};{dst.lng},{dst.lat}", client)

        async def _cand(c: DetourCandidate):
            total = await _mapbox_duration(
                f"{cur.lng},{cur.lat};{c.lng},{c.lat};{dst.lng},{dst.lat}", client)
            return c.id, total

        baseline, cand_pairs = await asyncio.gather(
            baseline_task, asyncio.gather(*[_cand(c) for c in candidates]))

    results: List[DetourResult] = []
    for cid, total in cand_pairs:
        added = None
        if baseline is not None and total is not None:
            added = max(0, total - baseline)
        results.append(DetourResult(id=cid, total_duration_sec=total, added_seconds=added))

    logger.info(f"along_route_detours: baseline={baseline} candidates={len(results)}")
    return DetoursResponse(baseline_duration_sec=baseline, results=results)


# ══════════════════════════════════════════════════════════════════════════════
#  Free-text search (any place) — Autocomplete + Place Details
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/search/autocomplete", response_model=List[AutocompleteItem])
async def search_autocomplete(q: str, lat: Optional[float] = None,
                              lng: Optional[float] = None):
    """
    Google Places Autocomplete, biased to Egypt (and toward lat/lng when given).
    Returns [] for empty input or any failure.
    """
    if not q or not q.strip():
        return []
    payload: dict = {"input": q.strip(), "includedRegionCodes": ["eg"]}
    if lat is not None and lng is not None:
        payload["locationBias"] = {"circle": {
            "center": {"latitude": lat, "longitude": lng}, "radius": DEFAULT_RADIUS_M}}

    field_mask = ("suggestions.placePrediction.placeId,"
                  "suggestions.placePrediction.structuredFormat")
    data = await _google_post(PLACES_AUTOCOMPLETE_URL, payload, field_mask)
    if not data:
        return []

    out: List[AutocompleteItem] = []
    for s in (data.get("suggestions") or []):
        pred = s.get("placePrediction") or {}
        pid = pred.get("placeId")
        if not pid:
            continue
        sf = pred.get("structuredFormat") or {}
        primary = ((sf.get("mainText") or {}).get("text") or "").strip()
        secondary = ((sf.get("secondaryText") or {}).get("text") or "").strip()
        primary = _fix_mojibake(primary, debug_label=f"autocomplete {pid} primary")
        secondary = _fix_mojibake(secondary, debug_label=f"autocomplete {pid} secondary")
        out.append(AutocompleteItem(place_id=pid, primary_text=primary or q.strip(),
                                    secondary_text=secondary))
    logger.info(f"search_autocomplete: q={q!r} hits={len(out)}")
    return out


@router.get("/search/place/{place_id}", response_model=Optional[PlaceDetails])
async def search_place_details(place_id: str):
    """
    Google Place Details for a chosen autocomplete result. Returns null on failure.
    """
    field_mask = ("displayName,location,formattedAddress,internationalPhoneNumber,"
                  "rating,regularOpeningHours")
    data = await _google_get(PLACES_DETAILS_URL + place_id, field_mask)
    if not data:
        return None
    loc = data.get("location") or {}
    lat, lng = loc.get("latitude"), loc.get("longitude")
    if lat is None or lng is None:
        return None
    opening = ((data.get("regularOpeningHours") or {}).get("weekdayDescriptions")) or None
    details_name = ((data.get("displayName") or {}).get("text") or "").strip() or "Place"
    return PlaceDetails(
        name=_fix_mojibake(details_name, debug_label=f"place/{place_id} name"),
        lat=float(lat), lng=float(lng),
        phone=data.get("internationalPhoneNumber"),
        rating=data.get("rating"),
        address=_fix_mojibake(data.get("formattedAddress"), debug_label=f"place/{place_id} address"),
        opening_hours=opening,
    )
