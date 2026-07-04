"""
services/route_geometry.py

Pure-geometry helpers for the "Add stop along the route" feature. No external
I/O — just the math the along-route POI search needs:

  • decode an encoded polyline6 string  → [(lat, lng), ...]
  • find the nearest route vertex to the user  → slice off the REMAINING route
  • cumulative along-route distances
  • project a POI onto the remaining route  → (perpendicular_m, along_route_m)
  • an expanded bounding box for the Overpass fallback

Method: everything is done with a haversine metric plus a local planar
point-to-segment projection (same approach already used in
services/mapbox_traffic.py). shapely is intentionally NOT a dependency — the
routes here are short (a single city trip) so the small-angle planar projection
for the perpendicular foot is accurate to well within the 800 m corridor
tolerance, and the leg lengths themselves use exact haversine.

Coordinate convention inside this module: (lat, lng) tuples. The public API
accepts GeoJSON [lng, lat] order and converts on the way in.
"""

from __future__ import annotations

import math
from typing import List, Tuple

LatLng = Tuple[float, float]

_R = 6_371_000.0  # Earth radius (m)


# ── distance primitives ───────────────────────────────────────────────────────
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres."""
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return 2 * _R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _project_param(a: LatLng, b: LatLng, p: LatLng,
                   clamp_low: bool = True, clamp_high: bool = True) -> float:
    """
    Projection parameter t of point p onto segment a→b, computed in a local
    equirectangular plane (lng scaled by cos(lat)) so it is distortion-aware
    near Cairo's latitude. By default clamped to [0,1] (nearest point on the
    finite segment).

    `clamp_low=False` lets t go negative — used only for the route's very
    first segment, so a point that is genuinely BEHIND the route start (not
    just near it) yields a negative t instead of being clamped to the start
    vertex. Without this, project_point() cannot distinguish "right at the
    start" (t=0) from "500 m behind the start" (both would otherwise read as
    t=0 → along_route_distance_m=0), and the "drop POIs behind the user"
    filter (which checks along < 0) would never trigger for such points.
    """
    lat0 = math.radians((a[0] + b[0]) / 2)
    cos0 = math.cos(lat0)
    ax, ay = a[1] * cos0, a[0]
    bx, by = b[1] * cos0, b[0]
    px, py = p[1] * cos0, p[0]
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    lo = 0.0 if clamp_low else float("-inf")
    hi = 1.0 if clamp_high else float("inf")
    return max(lo, min(hi, t))


# ── polyline6 decoding ────────────────────────────────────────────────────────
def decode_polyline6(encoded: str) -> List[LatLng]:
    """
    Decode a Google/Mapbox encoded polyline with precision 6 into [(lat, lng)].
    Mapbox Directions geometries use precision 6 (the app's route geometry).
    """
    coords: List[LatLng] = []
    index = lat = lng = 0
    length = len(encoded)
    factor = 1e6
    while index < length:
        for _shift_target in ("lat", "lng"):
            result = 1
            shift = 0
            while True:
                b = ord(encoded[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1F:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if _shift_target == "lat":
                lat += delta
            else:
                lng += delta
        coords.append((lat / factor, lng / factor))
    return coords


def geojson_to_latlng(coords: List[List[float]]) -> List[LatLng]:
    """GeoJSON [[lng, lat], ...] → [(lat, lng), ...], dropping malformed points."""
    out: List[LatLng] = []
    for c in coords or []:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            lng, lat = float(c[0]), float(c[1])
            out.append((lat, lng))
    return out


# ── route slicing / measuring ─────────────────────────────────────────────────
def nearest_vertex_index(route: List[LatLng], point: LatLng) -> int:
    """Index of the route vertex closest to `point` (straight nearest vertex)."""
    best_i, best_d = 0, float("inf")
    for i, (rlat, rlng) in enumerate(route):
        d = haversine_m(point[0], point[1], rlat, rlng)
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def remaining_route(route: List[LatLng], current: LatLng) -> List[LatLng]:
    """
    The portion of the route from the vertex nearest the user to the end.
    Everything the user has already driven past is dropped.
    """
    if not route:
        return []
    idx = nearest_vertex_index(route, current)
    return route[idx:]


def cumulative_distances(route: List[LatLng]) -> List[float]:
    """Along-route distance (m) at each vertex, starting at 0.0."""
    cum = [0.0]
    for i in range(1, len(route)):
        cum.append(cum[-1] + haversine_m(route[i - 1][0], route[i - 1][1],
                                         route[i][0], route[i][1]))
    return cum


def project_point(route: List[LatLng], cum: List[float], p: LatLng
                  ) -> Tuple[float, float]:
    """
    Project point p onto the polyline `route` (with precomputed cumulative
    distances `cum`). Returns (perpendicular_distance_m, along_route_distance_m).

    Chooses the segment whose perpendicular foot is closest to p, then reports
    that perpendicular distance and the cumulative distance of the foot from the
    route start. For a single-vertex route, falls back to a plain point distance.

    The first segment's LOW end is left unclamped (see _project_param) so a
    point genuinely behind the route start projects to a negative
    along_route_distance_m rather than collapsing to 0 — callers filtering on
    "along < 0" then correctly drop it as behind the user.
    """
    if len(route) < 2:
        if route:
            return haversine_m(p[0], p[1], route[0][0], route[0][1]), 0.0
        return float("inf"), 0.0

    best_perp = float("inf")
    best_along = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        t = _project_param(a, b, p, clamp_low=(i > 0))
        foot = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        perp = haversine_m(p[0], p[1], foot[0], foot[1])
        if perp < best_perp:
            best_perp = perp
            seg_len = cum[i + 1] - cum[i]
            best_along = cum[i] + t * seg_len
    return best_perp, best_along


def sample_along(route: List[LatLng], cum: List[float], spacing_m: float,
                 max_points: int) -> List[LatLng]:
    """
    Evenly-spaced anchor points along the route (~`spacing_m` apart), capped to
    `max_points`. These are the circle-search centres for the Google corridor
    scan. Always includes the start; if the route is longer than
    spacing_m * max_points the spacing is widened so the whole route is covered.
    """
    total = cum[-1] if cum else 0.0
    if total <= 0.0 or not route:
        return route[:1]
    n = int(total // spacing_m) + 1
    if n > max_points:
        n = max_points
        spacing_m = total / max(1, n - 1) if n > 1 else total
    targets = [min(total, i * spacing_m) for i in range(n)]

    pts: List[LatLng] = []
    j = 0
    for target in targets:
        while j < len(cum) - 1 and cum[j + 1] < target:
            j += 1
        if j >= len(route) - 1:
            pts.append(route[-1])
            continue
        seg_len = cum[j + 1] - cum[j]
        t = (target - cum[j]) / seg_len if seg_len > 0 else 0.0
        a, b = route[j], route[j + 1]
        pts.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return pts


def corridor_bbox(route: List[LatLng], buffer_m: float
                  ) -> Tuple[float, float, float, float]:
    """
    Bounding box of the route expanded by `buffer_m` on every side.
    Returns (min_lat, min_lng, max_lat, max_lng) — the (south, west, north, east)
    order Overpass expects.
    """
    lats = [p[0] for p in route]
    lngs = [p[1] for p in route]
    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)
    dlat = buffer_m / 111_320.0
    mid_lat = (min_lat + max_lat) / 2
    dlng = buffer_m / (111_320.0 * max(0.1, math.cos(math.radians(mid_lat))))
    return (min_lat - dlat, min_lng - dlng, max_lat + dlat, max_lng + dlng)
