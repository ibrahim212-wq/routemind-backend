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


# ══════════════════════════════════════════════════════════════════════════════
# LIVE TRAFFIC VECTOR TILES  (mapbox.mapbox-traffic-v1)
# ──────────────────────────────────────────────────────────────────────────────
# Per-junction live congestion from the SAME source the app's map renders, read
# from the Traffic Vector Tile (.mvt) — NOT the Directions API above (the
# Directions annotations only cover ~27% of Cairo roads; the tiles cover ~70-100%
# with the z13+z12 union, empirically verified 2026-06-17).
#
# Self-contained, ZERO new dependencies: the MVT protobuf is decoded by a small
# pure-Python reader (below) so we do NOT pull in mapbox-vector-tile / shapely /
# pyclipper / a second protobuf — which would risk a version clash with the
# protobuf firebase-admin already depends on (and break the FCM path on deploy).
# ══════════════════════════════════════════════════════════════════════════════

import math
import struct
import time

MAPBOX_SECRET_TOKEN = os.getenv("MAPBOX_SECRET_TOKEN", "")
_TRAFFIC_TILE_URL   = "https://api.mapbox.com/v4/mapbox.mapbox-traffic-v1/{z}/{x}/{y}.mvt"

# Mapbox traffic congestion label → jam on the 0..10 scanner scale (aligns with
# JAM_LEVELS in services/scanner.py: ADVISORY 1.5 / WARNING 3.2 / SERIOUS 5.5 / CRITICAL 8.0)
MAPBOX_CONG_TO_JAM10 = {"low": 1.5, "moderate": 4.0, "heavy": 7.0, "severe": 9.5}

# Confirmed in testing: the traffic layer uses extent 262144 (NOT the 4096 default).
# We still read the real extent from each tile; this is only the documented baseline.
_TRAFFIC_EXTENT_DEFAULT = 262144

_TILE_TTL         = 180     # tiles refresh every 30-60s; 3-min cache is safe & cheap
_TILE_BUDGET      = 30      # max NETWORK tile fetches per trip-alert request
_TILE_CONCURRENCY = 10      # cap concurrent tile fetches
_MATCH_LIMIT_M    = 200     # a road segment must pass within 200m of the junction
_TILE_ZOOMS       = (13, 12)  # try z13 (detail) first, fall back to z12 (coverage)

# Tile-level cache shared across requests: {(z, tx, ty): (decoded_layer, ts)}
_tile_cache: dict = {}
_tile_sem = None            # lazily created on the running loop


def _get_tile_sem():
    global _tile_sem
    if _tile_sem is None:
        _tile_sem = asyncio.Semaphore(_TILE_CONCURRENCY)
    return _tile_sem


def _tile_token() -> str:
    """Prefer the sk. secret token (server-side tiles); fall back to pk."""
    return os.getenv("MAPBOX_SECRET_TOKEN", "") or os.getenv("MAPBOX_TOKEN", "")


class _TileBudgetExceeded(Exception):
    """Raised when the per-request network tile-fetch budget is exhausted."""


# ── A1. lat/lon → tile x/y (Web Mercator slippy-map) ──────────────────────────
def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n  = 2 ** zoom
    x  = int((lon + 180.0) / 360.0 * n)
    lr = math.radians(lat)
    y  = int((1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n)
    # clamp into valid tile range
    x  = min(max(x, 0), n - 1)
    y  = min(max(y, 0), n - 1)
    return x, y


def _tile_bounds(tx: int, ty: int, zoom: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a tile — exact replica of mercantile.bounds."""
    n = 2 ** zoom

    def _ul(x: int, y: int) -> tuple[float, float]:
        lon = x / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
        return lon, lat

    west, north = _ul(tx, ty)
    east, south = _ul(tx + 1, ty + 1)
    return west, south, east, north


# ── A2. MVT tile-local pixel → WGS84 (full decode, NOT centroid) ──────────────
def tile_to_latlon(tx: int, ty: int, zoom: int, px: float, py: float,
                   extent: int = _TRAFFIC_EXTENT_DEFAULT) -> tuple[float, float]:
    """
    Convert an MVT integer coordinate (tile-local, 0..extent, origin top-left,
    y increasing downward) to WGS84 (lat, lon).
    """
    west, south, east, north = _tile_bounds(tx, ty, zoom)
    lon = west  + (px / extent) * (east  - west)
    lat = north - (py / extent) * (north - south)
    return lat, lon


# ── Pure-Python MVT (protobuf) decoder — zero external deps ───────────────────
def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift  = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _zigzag(n: int) -> int:
    return (n >> 1) ^ (-(n & 1))


def _skip_field(buf: bytes, pos: int, wire: int) -> int:
    if wire == 0:      # varint
        _, pos = _read_varint(buf, pos)
    elif wire == 1:    # 64-bit
        pos += 8
    elif wire == 2:    # length-delimited
        ln, pos = _read_varint(buf, pos)
        pos += ln
    elif wire == 5:    # 32-bit
        pos += 4
    else:
        raise ValueError(f"unsupported wire type {wire}")
    return pos


def _decode_value(buf: bytes) -> object:
    """Decode an MVT Value message → python scalar."""
    pos = 0
    end = len(buf)
    while pos < end:
        tag   = buf[pos]; pos += 1
        field = tag >> 3
        wire  = tag & 0x7
        if field == 1 and wire == 2:                       # string_value
            ln, pos = _read_varint(buf, pos)
            return buf[pos:pos + ln].decode("utf-8", "replace")
        if field == 2 and wire == 5:                       # float_value
            return struct.unpack_from("<f", buf, pos)[0]
        if field == 3 and wire == 1:                       # double_value
            return struct.unpack_from("<d", buf, pos)[0]
        if field == 4 and wire == 0:                       # int_value
            v, pos = _read_varint(buf, pos); return v
        if field == 5 and wire == 0:                       # uint_value
            v, pos = _read_varint(buf, pos); return v
        if field == 6 and wire == 0:                       # sint_value
            v, pos = _read_varint(buf, pos); return _zigzag(v)
        if field == 7 and wire == 0:                       # bool_value
            v, pos = _read_varint(buf, pos); return bool(v)
        pos = _skip_field(buf, pos, wire)
    return None


def _decode_geometry(geom: list[int]) -> list[list[tuple[int, int]]]:
    """
    Decode packed MVT geometry command integers → list of rings (each a list of
    (x, y) tile-local integer coords). Handles MoveTo(1)/LineTo(2)/ClosePath(7).
    """
    rings: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    x = y = 0
    i = 0
    n = len(geom)
    while i < n:
        cmd_int = geom[i]; i += 1
        cmd_id  = cmd_int & 0x7
        count   = cmd_int >> 3
        if cmd_id == 1:                       # MoveTo — starts a new ring/line
            for _ in range(count):
                if i + 1 >= n:
                    break
                x += _zigzag(geom[i]);     i += 1
                y += _zigzag(geom[i]);     i += 1
                if cur:
                    rings.append(cur)
                cur = [(x, y)]
        elif cmd_id == 2:                     # LineTo — extends current line
            for _ in range(count):
                if i + 1 >= n:
                    break
                x += _zigzag(geom[i]);     i += 1
                y += _zigzag(geom[i]);     i += 1
                cur.append((x, y))
        elif cmd_id == 7:                     # ClosePath — polygons (rare here)
            if cur:
                cur.append(cur[0])
        else:
            break
    if cur:
        rings.append(cur)
    return rings


def _decode_feature(buf: bytes, keys: list[str], values: list) -> dict:
    pos = 0
    end = len(buf)
    tags: list[int] = []
    geom: list[int] = []
    gtype = 0
    while pos < end:
        tag   = buf[pos]; pos += 1
        field = tag >> 3
        wire  = tag & 0x7
        if field == 2 and wire == 2:          # packed tags
            ln, pos = _read_varint(buf, pos)
            sub_end = pos + ln
            while pos < sub_end:
                v, pos = _read_varint(buf, pos)
                tags.append(v)
        elif field == 3 and wire == 0:        # geometry type
            gtype, pos = _read_varint(buf, pos)
        elif field == 4 and wire == 2:        # packed geometry
            ln, pos = _read_varint(buf, pos)
            sub_end = pos + ln
            while pos < sub_end:
                v, pos = _read_varint(buf, pos)
                geom.append(v)
        else:
            pos = _skip_field(buf, pos, wire)

    props = {}
    for j in range(0, len(tags) - 1, 2):
        ki, vi = tags[j], tags[j + 1]
        if 0 <= ki < len(keys) and 0 <= vi < len(values):
            props[keys[ki]] = values[vi]
    return {"properties": props, "type": gtype, "geometry": _decode_geometry(geom)}


def _decode_layer(buf: bytes) -> dict:
    pos = 0
    end = len(buf)
    name = ""
    extent = 4096
    feat_blobs: list[bytes] = []
    keys: list[str] = []
    values: list = []
    while pos < end:
        tag   = buf[pos]; pos += 1
        field = tag >> 3
        wire  = tag & 0x7
        if field == 1 and wire == 2:                       # name
            ln, pos = _read_varint(buf, pos)
            name = buf[pos:pos + ln].decode("utf-8", "replace"); pos += ln
        elif field == 2 and wire == 2:                     # feature
            ln, pos = _read_varint(buf, pos)
            feat_blobs.append(buf[pos:pos + ln]); pos += ln
        elif field == 3 and wire == 2:                     # key
            ln, pos = _read_varint(buf, pos)
            keys.append(buf[pos:pos + ln].decode("utf-8", "replace")); pos += ln
        elif field == 4 and wire == 2:                     # value
            ln, pos = _read_varint(buf, pos)
            values.append(_decode_value(buf[pos:pos + ln])); pos += ln
        elif field == 5 and wire == 0:                     # extent
            extent, pos = _read_varint(buf, pos)
        else:
            pos = _skip_field(buf, pos, wire)

    features = [_decode_feature(fb, keys, values) for fb in feat_blobs]
    # MVT raw geometry has origin top-left, y increasing DOWNWARD. Flip to the
    # y-up convention (y' = extent - y) so coordinates match the reference
    # mapbox-vector-tile library exactly (verified vertex-exact) and feed
    # tile_to_latlon correctly — confirmed against real Cairo roads 2026-06-17
    # (e.g. Corniche-on-the-Nile matches at 7m under this convention vs 159m raw).
    for f in features:
        f["geometry"] = [[(x, extent - y) for (x, y) in ring] for ring in f["geometry"]]
    return {"name": name, "extent": extent, "features": features}


def decode_mvt(raw: bytes) -> dict:
    """Decode a full MVT tile → {layer_name: {extent, features:[...]}}. Pure python."""
    pos = 0
    end = len(raw)
    layers: dict = {}
    while pos < end:
        tag   = raw[pos]; pos += 1
        field = tag >> 3
        wire  = tag & 0x7
        if field == 3 and wire == 2:          # Tile.layers
            ln, pos = _read_varint(raw, pos)
            layer = _decode_layer(raw[pos:pos + ln]); pos += ln
            layers[layer["name"]] = layer
        else:
            pos = _skip_field(raw, pos, wire)
    return layers


# ── Geo distance: point → polyline (haversine, segment-aware, NOT centroid) ───
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _point_to_segment_m(a_lat, a_lon, b_lat, b_lon, p_lat, p_lon) -> float:
    """Distance (m) from point P to segment A-B, projected in local planar approx."""
    dx, dy = (b_lon - a_lon), (b_lat - a_lat)
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return _haversine_m(p_lat, p_lon, a_lat, a_lon)
    t = ((p_lon - a_lon) * dx + (p_lat - a_lat) * dy) / seg2
    t = max(0.0, min(1.0, t))
    proj_lon = a_lon + t * dx
    proj_lat = a_lat + t * dy
    return _haversine_m(p_lat, p_lon, proj_lat, proj_lon)


def _min_dist_to_ring_m(p_lat, p_lon, ring_wgs84) -> float:
    if not ring_wgs84:
        return float("inf")
    if len(ring_wgs84) == 1:
        return _haversine_m(p_lat, p_lon, ring_wgs84[0][0], ring_wgs84[0][1])
    best = float("inf")
    for i in range(len(ring_wgs84) - 1):
        a_lat, a_lon = ring_wgs84[i]
        b_lat, b_lon = ring_wgs84[i + 1]
        d = _point_to_segment_m(a_lat, a_lon, b_lat, b_lon, p_lat, p_lon)
        if d < best:
            best = d
    return best


# ── A3. fetch one decoded traffic tile (cache + budget + semaphore) ───────────
async def fetch_tile(tx: int, ty: int, zoom: int,
                     client: "httpx.AsyncClient | None" = None,
                     budget: dict | None = None) -> tuple[dict, bool]:
    """
    Return (traffic_layer, from_cache). traffic_layer = {"extent", "features"}.
    Raises _TileBudgetExceeded if a network fetch is needed but the per-request
    budget is spent. Raises on non-200 / decode error (caller handles).
    """
    key = (zoom, tx, ty)
    now = time.time()
    hit = _tile_cache.get(key)
    if hit and (now - hit[1]) < _TILE_TTL:
        if budget is not None:
            budget["tile_cache_hits"] = budget.get("tile_cache_hits", 0) + 1
        return hit[0], True

    # Reserve the budget slot BEFORE the await — check-then-increment with no
    # await in between is atomic under asyncio, so concurrent enrich() coroutines
    # (run via asyncio.gather) cannot all slip past the cap. Incrementing AFTER
    # the network await would race and blow the budget.
    if budget is not None:
        if budget.get("tile_used", 0) >= budget.get("tile_max", _TILE_BUDGET):
            budget["tile_deferred"] = budget.get("tile_deferred", 0) + 1
            raise _TileBudgetExceeded()
        budget["tile_used"] = budget.get("tile_used", 0) + 1

    url   = _TRAFFIC_TILE_URL.format(z=zoom, x=tx, y=ty)
    token = _tile_token()
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=8.0)
    try:
        async with _get_tile_sem():
            resp = await client.get(url, params={"access_token": token}, timeout=8.0)
    finally:
        if owns_client:
            await client.aclose()

    if resp.status_code != 200:
        raise RuntimeError(f"traffic tile {zoom}/{tx}/{ty} → HTTP {resp.status_code}")

    raw = resp.content
    if raw[:2] == b"\x1f\x8b":              # gzip (httpx usually decompresses already)
        import gzip
        raw = gzip.decompress(raw)

    layers = decode_mvt(raw)
    layer  = layers.get("traffic") or {"extent": _TRAFFIC_EXTENT_DEFAULT, "features": []}
    _tile_cache[key] = (layer, now)
    return layer, False


# ── A4. live per-junction congestion ──────────────────────────────────────────
async def get_live_congestion(lat: float, lon: float,
                              client: "httpx.AsyncClient | None" = None,
                              budget: dict | None = None) -> dict:
    """
    Live congestion for one point from the Mapbox traffic tiles.

    Returns: {jam10, congestion_label, source, dist_m}
      - source "mapbox_tiles" with jam10 set when a real segment matches ≤200m
      - source "no_data"  (jam10=None) when no usable segment is found (→ caller
        may fall back to TomTom / patterns)
      - source "budget"   (jam10=None) when the per-request tile budget is spent
        (→ caller should fall back to patterns, NOT keep hitting live APIs)
    NEVER raises.
    """
    budget_hit = False
    best_label = None
    best_dist  = float("inf")

    for zoom in _TILE_ZOOMS:
        tx, ty = latlon_to_tile(lat, lon, zoom)
        try:
            layer, _from_cache = await fetch_tile(tx, ty, zoom, client=client, budget=budget)
        except _TileBudgetExceeded:
            budget_hit = True
            break
        except Exception as ex:
            logger.warning(f"traffic tile fetch failed z{zoom} ({lat},{lon}): {ex}")
            continue

        extent = layer.get("extent", _TRAFFIC_EXTENT_DEFAULT)
        for feat in layer.get("features", []):
            label = feat.get("properties", {}).get("congestion")
            if not label or label == "unknown":
                continue
            for ring in feat.get("geometry", []):
                ring_wgs = [tile_to_latlon(tx, ty, zoom, px, py, extent) for (px, py) in ring]
                d = _min_dist_to_ring_m(lat, lon, ring_wgs)
                if d < best_dist:
                    best_dist  = d
                    best_label = label
        if best_label is not None and best_dist <= _MATCH_LIMIT_M:
            break                              # good match at this zoom — stop

    if best_label is not None and best_dist <= _MATCH_LIMIT_M:
        return {
            "jam10":            MAPBOX_CONG_TO_JAM10.get(best_label, 0.0),
            "congestion_label": best_label,
            "source":           "mapbox_tiles",
            "dist_m":           round(best_dist),
        }
    if budget_hit:
        return {"jam10": None, "congestion_label": None, "source": "budget", "dist_m": None}
    return {"jam10": None, "congestion_label": None, "source": "no_data",
            "dist_m": round(best_dist) if best_dist != float("inf") else None}