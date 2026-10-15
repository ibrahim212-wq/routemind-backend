"""
api/places.py
Place search + "Add stop along the route" backend, on RouteMind's OWN place index.

Places come from services/place_index.py: gzip SQLite tiles of Egyptian open data (OpenStreetMap,
Overture, … — the same index and the same search engine the phones use), downloaded lazily and
cached on disk. No API key, no paid service. The data carries names, categories, addresses, phones,
websites and sometimes OSM opening hours — NOT ratings, reviews, prices or an open-now state, so
those fields are always null and filters on them (min_rating, open_now) cannot be honoured.
Endpoints (kept for older app builds that still call them):

  POST /api/places/nearby            — nearby by category / free-text
  POST /api/along-route/pois         — category POIs along the REMAINING route
  POST /api/along-route/detours      — extra driving time to visit each candidate (Mapbox)
  GET  /api/search/autocomplete      — place search suggestions (any place)
  GET  /api/search/place/{place_id}  — details for a chosen result (id "rm:<pid>")

Detours reuse the Mapbox Directions pattern + MAPBOX_TOKEN from services/mapbox_traffic.py (read
from env only, never returned or logged). Corridor geometry lives in services/route_geometry.py;
the OSM/Overpass fallback (used when the index has nothing for a corridor) in services/overpass.py.

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
  curl -s 'localhost:8080/api/search/place/rm:PID_HERE'
"""

import os
import asyncio
import logging
from typing import Optional, List

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from services import route_geometry as geo
from services import overpass
from services import place_index
from services.place_index import CATEGORY_WORDS

logger = logging.getLogger("routemind.places")
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────

MAPBOX_TOKEN      = os.getenv("MAPBOX_TOKEN", "")   # same var as mapbox_traffic.py
MAPBOX_DIRECTIONS_URL   = "https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords}"

DEFAULT_RADIUS_M = 5000.0   # kept for search_places' signature (the engine sizes its own search area)
MAX_LIMIT        = 10
NEAREST_FETCH    = 20       # category "nearest" asks: candidates fetched before the distance sort
CORRIDOR_FETCH   = int(os.getenv("CORRIDOR_FETCH", "60"))   # engine results before the corridor filter
DEFAULT_ANCHOR   = (30.0444, 31.2357)   # Tahrir — autocomplete without a position

# ── Along-route tunables (env-overridable so they can be tuned in Cloud Run) ────
CORRIDOR_BUFFER_M    = float(os.getenv("CORRIDOR_BUFFER_M", "1500"))  # Overpass bbox pad
CORRIDOR_WIDTH_M     = float(os.getenv("CORRIDOR_WIDTH_M", "800"))    # perpendicular filter
MAX_DETOUR_CANDIDATES = int(os.getenv("MAX_DETOUR_CANDIDATES", "8"))
MAX_CORRIDOR_ANCHORS  = int(os.getenv("MAX_CORRIDOR_ANCHORS", "12"))
# Two-band corridor anchoring (corridor_anchors): a gapless near band over the
# first ~18 km + a sparse far band. The index search takes the whole remaining
# route instead; kept for callers that still want sample points along a route.
CORRIDOR_NEAR_SPACING_M = float(os.getenv("CORRIDOR_NEAR_SPACING_M", "1500"))
CORRIDOR_NEAR_ANCHORS   = int(os.getenv("CORRIDOR_NEAR_ANCHORS", "12"))
CORRIDOR_FAR_ANCHORS    = int(os.getenv("CORRIDOR_FAR_ANCHORS", "8"))


def corridor_anchors(remaining, cum):
    """Sample points along the remaining route: gapless near band (first
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

# /places/nearby's category vocabulary: any key of CATEGORY_WORDS (gas_station,
# pharmacy, atm, restaurant, cafe, and fuel / parking / hospital / mosque).
# Along-route: the app's 5 stop categories (the Overpass fallback knows these).
_ALONG_ROUTE_CATEGORIES = ("fuel", "restaurant", "cafe", "atm", "parking")

# ── Models ──────────────────────────────────────────────────────────────────--

class NearbyRequest(BaseModel):
    lat:      float
    lng:      float
    category: Optional[str] = None   # gas_station|pharmacy|atm|restaurant|cafe|…
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

def _category_word(category: Optional[str]) -> Optional[str]:
    """A stop category / Google-style place type → the engine's word for it."""
    return CATEGORY_WORDS.get((category or "").strip().lower())


def _decode_polyline5(encoded: str) -> List[tuple]:
    """Precision-5 encoded polyline (what geo.encode_polyline5 writes) → [(lat, lng)]."""
    out: List[tuple] = []
    index = lat = lng = 0
    try:
        while index < len(encoded):
            vals = []
            for _ in range(2):
                shift = result = 0
                while True:
                    b = ord(encoded[index]) - 63
                    index += 1
                    result |= (b & 0x1F) << shift
                    shift += 5
                    if b < 0x20:
                        break
                vals.append(~(result >> 1) if result & 1 else result >> 1)
            lat += vals[0]
            lng += vals[1]
            out.append((lat / 1e5, lng / 1e5))
    except IndexError:
        pass   # a truncated polyline: keep the points decoded so far
    return out


def _to_place_result(p: dict) -> PlaceResult:
    return PlaceResult(name=p["name"], lat=p["lat"], lng=p["lng"], address=p.get("address"),
                       distance_m=int(p.get("distance_m") or 0))

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
    Resolve nearby places from the own index — category (nearest first) or
    free-text (the engine's own order: name match, distance, prominence).
    Returns ≤ limit results. NEVER raises — on any failure (unknown category,
    no index yet) returns [].

    included_type (free-text mode only) constrains the search to one category
    when it has an engine word (gas_station, pharmacy, atm, restaurant, cafe,
    parking, hospital, mosque): query "Master" + "gas_station" searches
    «بنزينة Master» — fuel stations named Master, not any shop with the word.
    """
    n = max(1, min(MAX_LIMIT, limit or 1))

    if query and query.strip():
        found = await place_index.find(query.strip(), lat, lng, limit=n,
                                       kind_word=_category_word(included_type))
        return [_to_place_result(p) for p in found][:n]

    word = _category_word(category)
    if word is None:
        logger.warning(f"resolve_nearby: missing/unknown category={category!r}")
        return []
    found = await place_index.find(word, lat, lng, limit=max(n, NEAREST_FETCH))
    found.sort(key=lambda p: p.get("distance_m") or 0)   # nearest first
    return [_to_place_result(p) for p in found][:n]

# ══════════════════════════════════════════════════════════════════════════════
#  The copilot's place retrieval
#
#  Open data: names, categories, addresses, phones, websites, sometimes hours.
#  rating / reviews / price / open_now are always None — never invented.
# ══════════════════════════════════════════════════════════════════════════════

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
    route_polyline: Optional[str] = None,  # encoded polyline5 → search along that route
    limit: int = 5,
) -> List[dict]:
    """
    Free-text place search on the own index around (lat, lng), or along the
    route when route_polyline is given. Returns place dicts (services/
    place_index: id "rm:<pid>", name, address, category, distance_m, phone,
    website, hours, and rating/reviews/price/open_now = None). NEVER raises → [].

    Not honoured, by construction of open data: min_rating (there are no
    ratings — nothing is filtered by it) and open_now (no open state; passed
    through without filtering, every result keeps open_now None). radius_m is
    a bias only, as before: the engine sizes its own search area.
    """
    q = (query or "").strip()
    if not q:
        return []
    n = max(1, min(20, limit))
    route = _decode_polyline5(route_polyline) if route_polyline else None
    found = await place_index.find(q, lat, lng, limit=n, route=route or None,
                                   kind_word=_category_word(included_type))
    if rank == "distance":
        found.sort(key=lambda p: p.get("distance_m") or 0)
    return found[:n]


async def place_details_rich(place_id: str) -> Optional[dict]:
    """One place by id ("rm:<pid>" or the bare pid) with what the open data has:
    phone, website, hours (the raw OSM opening_hours string, as a one-item
    list) — no reviews, no editorial summary. None when unknown."""
    if not place_id:
        return None
    base = await place_index.by_pid(place_id)
    if not base:
        return None
    hours = base.get("hours")
    base.update({
        "editorial": None,
        "hours": [hours] if hours else None,
        "review_samples": [],
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
    source: str                                     # "index" | "overpass" | "none"
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
    rating: Optional[float] = None                  # always null: open data has no ratings
    address: Optional[str] = None
    opening_hours: Optional[List[str]] = None


# ── Along-route POIs: own index along the route (primary) + Overpass (fallback) ─

def _parse_route(req: "AlongRoutePoisRequest") -> List:
    """Route input → [(lat,lng), ...] from GeoJSON coords or an encoded polyline6."""
    if req.route:
        return geo.geojson_to_latlng(req.route)
    if req.polyline6:
        return geo.decode_polyline6(req.polyline6)
    return []


def _poi_dict(p: dict) -> dict:
    """An index place → the raw POI shape _filter_to_corridor takes (same as overpass.fetch_pois)."""
    return {"id": p["id"], "name": p.get("name") or "Place", "lat": p["lat"], "lng": p["lng"],
            "brand": None, "phone": p.get("phone"), "opening_hours": p.get("hours"),
            "address": p.get("address")}


async def _google_corridor_pois(gtype: str, anchors: List, n_per_anchor: int = 20
                                ) -> List[dict]:
    """
    Category POIs along a route, from the own index (the name is historical —
    api/copilot.py imports it). `gtype` is a stop category ("fuel") or a place
    type ("gas_station"); `anchors` is the route ahead as [(lat, lng), ...] —
    pass the remaining route itself (corridor_anchors() sample points work too).
    One engine search of the category word with route=anchors, from the first
    point. Returns raw POI dicts (id/name/lat/lng/brand/phone/opening_hours/
    address) for _filter_to_corridor. Empty on failure. n_per_anchor is unused.
    """
    word = _category_word(gtype)
    pts = [(float(a[0]), float(a[1])) for a in (anchors or [])]
    if word is None or not pts:
        return []
    found = await place_index.find(word, pts[0][0], pts[0][1], limit=CORRIDOR_FETCH,
                                   route=pts if len(pts) >= 2 else None)
    return [_poi_dict(p) for p in found]


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
    first. The own place index is the primary source; if it has nothing for the
    corridor we fall back to OSM/Overpass. Always HTTP 200 — empty on any failure.
    """
    category = (req.category or "").lower()
    word = _category_word(category) if category in _ALONG_ROUTE_CATEGORIES else None
    if word is None:
        logger.warning(f"along_route_pois: unknown category={req.category!r}")
        return AlongRoutePoisResponse(source="none", results=[])

    # NEAREST-TO-ME mode: closest of the category to the user's current location,
    # regardless of route; along_route_distance_m carries the straight-line
    # distance so the client's cards/sort work unchanged.
    if req.nearest:
        cl = req.current_location
        found = await place_index.find(word, cl.lat, cl.lng,
                                       limit=max(NEAREST_FETCH, req.max_results))
        out = [AlongRoutePoi(
            id=p["id"], name=p.get("name") or "Place", lat=p["lat"], lng=p["lng"],
            category=category, distance_from_route_m=int(p.get("distance_m") or 0),
            along_route_distance_m=int(p.get("distance_m") or 0),
            phone=p.get("phone"), opening_hours=p.get("hours"), address=p.get("address"))
            for p in found]
        out.sort(key=lambda r: r.along_route_distance_m)
        logger.info(f"along_route_pois NEAREST: category={category} hits={len(out)}")
        return AlongRoutePoisResponse(source="index" if out else "none",
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

    # 1) own index (primary): the category along the remaining route
    raw = await _google_corridor_pois(category, remaining)
    results = _filter_to_corridor(raw, remaining, cum, category, req.max_results)
    source = "index"
    logger.info(f"along_route_pois: category={category} index_raw={len(raw)} "
                f"index_after_filter={len(results)}")

    # 2) Overpass (fallback) — only when the index yielded nothing for this corridor
    if not results:
        bbox = geo.corridor_bbox(remaining, CORRIDOR_BUFFER_M)
        osm_raw = await overpass.fetch_pois(category, bbox)
        results = _filter_to_corridor(osm_raw, remaining, cum, category, req.max_results)
        source = "overpass" if results else "none"

    logger.info(f"along_route_pois: category={category} source={source} hits={len(results)}")
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
#  Free-text search (any place) — suggestions + details, on the own index
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/search/autocomplete", response_model=List[AutocompleteItem])
async def search_autocomplete(q: str, lat: Optional[float] = None,
                              lng: Optional[float] = None):
    """
    Place suggestions from the own index, around lat/lng when given (else
    central Cairo). place_id is "rm:<pid>" (→ /search/place/{place_id}).
    Returns [] for empty input or any failure.
    """
    if not q or not q.strip():
        return []
    alat, alng = (lat, lng) if lat is not None and lng is not None else DEFAULT_ANCHOR
    found = await place_index.find(q.strip(), alat, alng, limit=5)
    out = [AutocompleteItem(place_id=p["id"], primary_text=p.get("name") or q.strip(),
                            secondary_text=p.get("address") or "")
           for p in found]
    logger.info(f"search_autocomplete: q={q!r} hits={len(out)}")
    return out


@router.get("/search/place/{place_id}", response_model=Optional[PlaceDetails])
async def search_place_details(place_id: str):
    """
    Details for a chosen autocomplete result ("rm:<pid>"). Returns null when unknown.
    """
    p = await place_index.by_pid(place_id)
    if not p:
        return None
    return PlaceDetails(
        name=p.get("name") or "Place",
        lat=p["lat"], lng=p["lng"],
        phone=p.get("phone"),
        rating=None,
        address=p.get("address"),
        opening_hours=[p["hours"]] if p.get("hours") else None,
    )
