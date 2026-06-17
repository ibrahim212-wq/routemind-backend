"""
api/trip_alert.py
GET /api/trip-alert/{trip_id}

يرجع كل البيانات اللي الـ AlertMapScreen محتاجها:
  - trip details (origin, dest, leave_by)
  - alert junctions (lat/lng + level + eta)
"""

from fastapi import APIRouter, HTTPException
from model.loader import ModelLoader
from services.supabase_client import get_supabase

from datetime import datetime, timezone
import time
import logging
logger = logging.getLogger("routemind.trip_alert")

router = APIRouter()

# ─────────────────────────────────────────────────────────────────
# Phase 1 — per-spot enrichment (PATTERNS) + Google delay split
# Phase 2 — LIVE per-junction levels for near-horizon junctions (≤60 min) from
#           Mapbox Traffic Tiles (PRIMARY — same source the app map renders),
#           with TomTom flow as secondary, patterns as tertiary, and a neutral
#           tail. Shared tile/reading caches + per-request fetch budgets.
# Read-time only. No scan/firing/headline/DB-write changes. Google stays the
# magnitude anchor (total delay unchanged; live only changes per-junction LEVEL).
# ─────────────────────────────────────────────────────────────────

# Greater Cairo / Giza bounding box (approx). Outside → route-level only.
_CAIRO_BBOX        = (29.70, 30.35, 30.80, 31.60)   # lat_min, lat_max, lon_min, lon_max
_DIST_TO_ROUTE_KM  = 0.6                            # keep junctions this close to the polyline
_WARN_LEVELS       = ("WARNING", "SERIOUS", "CRITICAL")

# Phase 2 live-traffic config
_LIVE_HORIZON_MIN    = 60    # junctions reached within this many min → live; else patterns
_LIVE_TTL            = 240   # shared TomTom reading cache TTL (4 min) — keyed by junction_id
_MAPBOX_TILE_BUDGET  = 30    # max NETWORK Mapbox tile fetches per trip-alert request
_TOMTOM_BUDGET       = 25    # max TomTom flow calls per trip-alert request

# Shared across requests: {junction_id: ({jam01,speed,confidence}, ts)}
_live_cache: dict = {}


def _in_cairo(lat: float, lon: float) -> bool:
    la0, la1, lo0, lo1 = _CAIRO_BBOX
    return la0 <= lat <= la1 and lo0 <= lon <= lo1


def _pattern_jam10(jid: str, arrival_dt) -> float:
    """
    PATTERN-ONLY predicted jam (0..10 scanner scale) at arrival_dt:
    blend of the historical lookup and Prophet seasonal features. No live data.
    Reuses the scanner's existing pattern functions (which handle the UTC→Cairo
    conversion); does NOT modify tier1 / jam_to_level. This is the tertiary
    source in the Phase 2 chain (Mapbox → TomTom → patterns → neutral).
    """
    from services.scanner import _get_historical_prediction, _get_prophet_prediction
    vals = []
    h = _get_historical_prediction(jid, arrival_dt)
    if h: vals.append(h[0])
    p = _get_prophet_prediction(jid, arrival_dt)
    if p: vals.append(p[0])
    if not vals:
        return 0.0
    # Clamp to [0,10] — Prophet's seasonal component can dip below baseline
    # (negative) off-peak; mirror the scanner's _ensemble max(0.0,…) guard.
    return round(max(0.0, min(10.0, sum(vals) / len(vals))), 2)


async def get_live_traffic(junction_id, lat, lon, client, sem, budget):
    """
    SECONDARY live provider seam (used when Mapbox tiles return no_data). TomTom
    flowSegment via tomtom.get_junction_reading, with a shared TTL cache and a
    per-request call budget. Applies the cur≠ff guard: TomTom returns
    currentSpeed == freeFlowSpeed as its default when there is no probe data, so
    an exact match is treated as no_data (→ None → caller falls back to patterns).
    Returns {jam01, speed, confidence, source} or None.
    """
    from services.tomtom import get_junction_reading
    import asyncio

    now = time.time()
    hit = _live_cache.get(junction_id)
    if hit and (now - hit[1]) < _LIVE_TTL:
        return {**hit[0], "source": "live-cache"}      # cache hit → zero API cost

    if budget["tom_used"] >= budget["tom_max"]:
        budget["tom_deferred"] += 1
        return None                                    # budget exhausted → caller uses patterns

    budget["tom_used"] += 1                             # reserve before the await (atomic in asyncio)
    reading = None
    for attempt in range(2):                           # 1 try + 1 retry (covers transient / 429)
        async with sem:
            reading = await get_junction_reading(junction_id, lat, lon, client)
        if reading is not None:
            break
        await asyncio.sleep(1.5 ** attempt)
    if reading is None:
        return None

    # cur≠ff guard: exact equality == TomTom's no-probe default → treat as no_data.
    cur = reading.get("current_speed")
    ff  = reading.get("free_flow_speed")
    try:
        if cur is not None and ff is not None and float(cur) == float(ff):
            return None
    except (TypeError, ValueError):
        pass

    data = {
        "jam01":      round(min(1.0, max(0.0, float(reading["jam_factor"]) / 10.0)), 3),
        "speed":      reading.get("current_speed"),
        "confidence": reading.get("confidence"),
    }
    _live_cache[junction_id] = (data, now)
    return {**data, "source": "live"}


def _blend_live(live01: float, jid: str, arrival_dt) -> float:
    """
    Live-snapshot blend (plan weighting): 0.5·live + 0.25·Prophet + 0.25·historical.
    Missing pattern components → weights renormalized. Reuses the scanner's
    pattern functions (UTC→Cairo aware); does NOT modify tier1 / jam_to_level.
    """
    from services.scanner import _get_historical_prediction, _get_prophet_prediction
    parts = [(0.5, max(0.0, min(1.0, live01)))]
    h = _get_historical_prediction(jid, arrival_dt)
    if h: parts.append((0.25, max(0.0, min(10.0, h[0])) / 10.0))
    p = _get_prophet_prediction(jid, arrival_dt)
    if p: parts.append((0.25, max(0.0, min(10.0, p[0])) / 10.0))
    wsum = sum(w for w, _ in parts)
    return max(0.0, min(1.0, sum(w * v for w, v in parts) / wsum))


def _log_once(flags: dict, key: str, msg: str) -> None:
    """Emit a fallback/diagnostic log at most once per trip-alert request."""
    if key not in flags:
        flags[key] = True
        logger.info(msg)


async def _predict_jam10(jid, arrival_dt, lead_minutes, *, lat, lng,
                         client, sem, budget, flags):
    """
    LIVE-AWARE per-junction jam (0..10 scanner scale) + source tag.

    Decision (Phase 2 — Component B):
      lead ≤ _LIVE_HORIZON_MIN and Mapbox tiles return real data →
          jam10 = 0.5·live + 0.25·Prophet + 0.25·historical   source="live_mapbox"
      lead ≤ horizon and Mapbox no_data → TomTom flow (cur≠ff) →
          jam10 = 0.5·tomtom + 0.25·Prophet + 0.25·historical source="live_tomtom"
      lead ≤ horizon and Mapbox tile budget spent → patterns    source="pattern"
      lead ≤ horizon and both live sources miss → patterns      source="pattern"
      lead > horizon → patterns (Phase 1 path, unchanged)       source="pattern"
      nothing anywhere → neutral (no color/bubble)              source="none"

    Fallback chain (B3): Mapbox → TomTom → patterns → neutral. NEVER raises;
    NEVER returns negative (B4). Blend math reuses _blend_live (0.5/0.25/0.25 on
    the 0..1 scale, weights renormalized if a pattern component is missing).
    Returns (jam10: float, source: str, extra: dict).
    """
    extra = {"live_speed": None, "confidence": None,
             "congestion_label": None, "dist_m": None}

    def _patterns(src="pattern"):
        return round(max(0.0, min(10.0, _pattern_jam10(jid, arrival_dt))), 2), src, extra

    # > horizon → patterns only (Phase 1 behavior, unchanged)
    if lead_minutes > _LIVE_HORIZON_MIN:
        return _patterns()

    # ── 1) Mapbox traffic tiles (PRIMARY live source) ──
    mb = None
    try:
        from services.mapbox_traffic import get_live_congestion
        mb = await get_live_congestion(lat, lng, client=client, budget=budget)
    except Exception as ex:                            # never let live crash a junction
        _log_once(flags, "mapbox_err", f"per-spot: Mapbox tile layer error ({ex}) — falling back")

    if mb and mb.get("jam10") is not None:
        jam01 = _blend_live(mb["jam10"] / 10.0, jid, arrival_dt)
        extra["congestion_label"] = mb.get("congestion_label")
        extra["dist_m"]           = mb.get("dist_m")
        return round(max(0.0, min(10.0, jam01 * 10.0)), 2), "live_mapbox", extra

    # Mapbox tile budget exhausted → patterns (A4: stop fetching live this request)
    if mb and mb.get("source") == "budget":
        _log_once(flags, "tile_budget",
                  "per-spot: Mapbox tile budget hit — remaining junctions use patterns")
        return _patterns()

    _log_once(flags, "mapbox_miss", "per-spot: Mapbox no_data on ≥1 junction — trying TomTom")

    # ── 2) TomTom flow (SECONDARY live source; cur≠ff guard inside wrapper) ──
    tom = None
    try:
        tom = await get_live_traffic(jid, lat, lng, client, sem, budget)
    except Exception as ex:
        _log_once(flags, "tomtom_err", f"per-spot: TomTom error ({ex}) — falling back to patterns")

    if tom is not None:
        jam01 = _blend_live(tom["jam01"], jid, arrival_dt)
        extra["live_speed"] = tom.get("speed")
        extra["confidence"] = tom.get("confidence")
        return round(max(0.0, min(10.0, jam01 * 10.0)), 2), "live_tomtom", extra

    _log_once(flags, "tomtom_miss", "per-spot: TomTom no_data on ≥1 junction — using patterns")

    # ── 3) Patterns ──
    pj = _pattern_jam10(jid, arrival_dt)
    if pj > 0:
        return round(max(0.0, min(10.0, pj)), 2), "pattern", extra

    # ── 4) Neutral — no data anywhere → no color/bubble ──
    return 0.0, "none", extra


def _distribute_delay(spots: list, total_delay: int) -> None:
    """
    Distribute the REAL Google delay across hotspots so the per-spot bubbles
    SUM EXACTLY to total_delay. Weight = jam01 for hotspots (level ≥ WARNING),
    else 0. Integer-rounded with the remainder pushed to the largest weight.
    Reconciliation: if there is real delay but no pattern hotspot, attribute it
    to the busiest on-route junction (never drop the delay, never fabricate).
    """
    for s in spots:
        s["delay_min"] = 0
    if total_delay <= 0:
        return
    weighted = [(s, s["jam_factor"] / 10.0) for s in spots if s["level"] in _WARN_LEVELS]
    wsum = sum(w for _, w in weighted)
    if wsum <= 0:
        if spots:                                   # real delay, no hotspot → busiest spot
            max(spots, key=lambda s: s["jam_factor"])["delay_min"] = total_delay
        return
    raw = [(s, total_delay * w / wsum) for s, w in weighted]
    for s, r in raw:
        s["delay_min"] = int(r)                     # floor
    remainder = total_delay - sum(s["delay_min"] for s, _ in raw)
    for s, _ in sorted(raw, key=lambda x: x[1], reverse=True):
        if remainder <= 0:
            break
        s["delay_min"] += 1
        remainder -= 1


def _normalize_alert_junctions(spots: list) -> None:
    """
    Shape-compatibility guard: guarantee EVERY alert_junction has the full field
    set with null-safe defaults, so the from_scan and fallback paths return an
    IDENTICAL shape the Flutter client can always deserialize. Required numeric
    fields never null (→ 0/0.0); required string fields never null (→ ""); only
    the optional Phase-3 fields (feeder_pressure/feeders/window/lead_minutes) may
    be null. Mutates in place.
    """
    for s in spots:
        dr = s.get("distance_to_route_km")
        dr = 0.0 if dr is None else float(dr)
        s["distance_to_route_km"]    = round(dr, 3)
        s["dist_to_segment_m"]       = round(dr * 1000.0, 1)   # client-expected alias
        dfo = s.get("distance_from_origin_km")
        s["distance_from_origin_km"] = 0.0 if dfo is None else round(float(dfo), 3)
        s["jam_factor"]   = 0.0 if s.get("jam_factor")  is None else float(s["jam_factor"])
        s["eta_minutes"]  = 0.0 if s.get("eta_minutes") is None else float(s["eta_minutes"])
        s["delay_min"]    = int(s.get("delay_min") or 0)
        s["level"]        = s.get("level") or "FREE"
        s["source"]       = s.get("source") or ""
        s["live_speed"]   = 0.0 if s.get("live_speed")  is None else float(s["live_speed"])
        s["confidence"]   = 0.0 if s.get("confidence")  is None else float(s["confidence"])
        s["congestion_label"] = s.get("congestion_label") or ""
        s["lat"]          = 0.0 if s.get("lat") is None else float(s["lat"])
        s["lng"]          = 0.0 if s.get("lng") is None else float(s["lng"])
        # optional / new Phase-3 fields — may be null, but key must exist
        s["feeders"]        = s.get("feeders") or []
        s.setdefault("feeder_pressure", None)
        s.setdefault("window", None)
        s.setdefault("lead_minutes", None)


async def _build_per_spot_enrichment(trip: dict, expected_delay_min: int, route_seconds: float):
    """
    Returns (alert_junctions, coverage). Per-junction LEVEL = LIVE TomTom (blended
    with patterns) for junctions reached within _LIVE_HORIZON_MIN, patterns beyond,
    with a per-junction fallback chain (TomTom → Google segment → patterns →
    neutral), a shared cache, and a per-request TomTom budget. The real Google
    delay is distributed across hotspots (bubbles sum exactly to expected_delay_min).
    """
    import asyncio
    import httpx
    from model.loader import ModelLoader
    from services.scanner import _jam_to_level, _estimate_etas, _parse_dt
    from services.junction_eta import compute_junction_etas

    meta460 = ModelLoader.get_meta()
    jids    = trip.get("junction_ids") or []

    # on-route TRAINED junctions, in order, deduped, with coords
    coords, seen, ordered = {}, set(), []
    for jid in jids:
        if jid in seen:
            continue
        row = meta460[meta460["junction_id"] == jid]
        if row.empty:
            continue
        seen.add(jid)
        r = row.iloc[0]
        coords[jid] = (float(r["latitude"]), float(r["longitude"]), int(r["junction_idx"]))
        ordered.append(jid)

    def cov(n, pct, poly, note, extra=None):
        c = {"trained_junctions_on_route": n, "route_coverage_pct": pct,
             "polyline_used": poly, "note": note,
             # Component D — live/source accounting (defaults for early returns)
             "live_junctions": 0, "pattern_junctions": 0, "no_data_junctions": 0,
             "primary_source": "patterns", "tile_cache_hits": 0,
             "source_breakdown": {}, "tomtom_calls": 0,
             # legacy keys the Flutter client still expects — keep, never remove
             "tomtom_deferred": 0, "google_fallback_calls": 0,
             "from_scan_results": False}
        if extra:
            c.update(extra)
        return c

    # bbox check — outside Greater Cairo/Giza → route-level only
    try:
        o_lat, o_lng = float(trip["origin_lat"]), float(trip["origin_lng"])
        d_lat, d_lng = float(trip["dest_lat"]),   float(trip["dest_lng"])
        outside = not (_in_cairo(o_lat, o_lng) or _in_cairo(d_lat, d_lng))
    except (KeyError, TypeError, ValueError):
        outside = False
    if outside:
        return [], cov(0, 0, False,
                       "Trip is outside the Cairo/Giza coverage area — route-level "
                       "delay only; spot-level data unavailable here.")
    if not ordered:
        return [], cov(0, 0, False,
                       "No trained junctions on this route — route-level delay only.")

    leave_by = _parse_dt(trip["leave_by"])
    dur      = route_seconds if route_seconds and route_seconds > 0 else 1800.0
    polyline = trip.get("route_polyline") or []

    # ── ETA list (no jam yet) — polyline preferred, straight-line fallback ──
    raw, polyline_used = [], False
    if isinstance(polyline, list) and len(polyline) >= 2:
        try:
            pl  = [(float(p[0]), float(p[1])) for p in polyline]
            jor = [{"junction_id": j, "junction_idx": coords[j][2],
                    "latitude": coords[j][0], "longitude": coords[j][1]} for j in ordered]
            for e in compute_junction_etas(pl, jor, dur, leave_by):
                if (e.get("distance_to_route_km") is not None
                        and e["distance_to_route_km"] > _DIST_TO_ROUTE_KM):
                    continue
                jid = e["junction_id"]
                raw.append({"jid": jid, "lat": coords[jid][0], "lng": coords[jid][1],
                            "eta_seconds": e["eta_seconds"], "eta_datetime": e["eta_datetime"],
                            "dist_orig": e.get("distance_from_origin_km"),
                            "dist_route": e.get("distance_to_route_km")})
            polyline_used = True
        except Exception as ex:
            logger.warning(f"per-spot polyline path failed ({ex}) — straight-line fallback")
            raw = []
    if not polyline_used:
        for e in _estimate_etas(ordered, dur, leave_by):
            raw.append({"jid": e["junction_id"], "lat": float(e["latitude"]),
                        "lng": float(e["longitude"]), "eta_seconds": e["eta_seconds"],
                        "eta_datetime": e["eta_datetime"],
                        "dist_orig": e.get("distance_from_origin_km"), "dist_route": None})

    # ── per-junction LEVEL: live (≤horizon) or patterns, with fallback chain ──
    # Component C: lead_minutes drives the live-vs-pattern decision. A junction
    # already in the past (departed-grace) → lead 0 → treated as live (now).
    now    = datetime.now(timezone.utc)
    budget = {"tile_used": 0, "tile_max": _MAPBOX_TILE_BUDGET,
              "tile_cache_hits": 0, "tile_deferred": 0,
              "tom_used": 0, "tom_max": _TOMTOM_BUDGET, "tom_deferred": 0}
    flags  = {}                                        # log-each-fallback-once latch
    sem    = asyncio.Semaphore(5)

    async def enrich(e, client):
        jid, arrival = e["jid"], e["eta_datetime"]
        lead_minutes = max(0.0, (arrival - now).total_seconds() / 60.0)
        jam10, source, extra = await _predict_jam10(
            jid, arrival, lead_minutes,
            lat=e["lat"], lng=e["lng"],
            client=client, sem=sem, budget=budget, flags=flags)
        return {"junction_id": jid, "lat": e["lat"], "lng": e["lng"],
                "level": _jam_to_level(jam10), "jam_factor": jam10,
                "eta_minutes": round(e["eta_seconds"] / 60, 1),
                "distance_from_origin_km": e["dist_orig"],
                "distance_to_route_km": e["dist_route"],
                "source": source,
                "live_speed": extra.get("live_speed"),
                "confidence": extra.get("confidence"),
                "congestion_label": extra.get("congestion_label"),
                "dist_to_segment_m": extra.get("dist_m")}

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            spots = list(await asyncio.gather(*[enrich(e, client) for e in raw]))
    except Exception as ex:                            # whole live layer down → patterns only
        logger.warning(f"live enrichment failed ({ex}) — patterns only")
        spots = []
        for e in raw:
            jam10 = _pattern_jam10(e["jid"], e["eta_datetime"])
            spots.append({"junction_id": e["jid"], "lat": e["lat"], "lng": e["lng"],
                          "level": _jam_to_level(jam10), "jam_factor": jam10,
                          "eta_minutes": round(e["eta_seconds"] / 60, 1),
                          "distance_from_origin_km": e["dist_orig"],
                          "distance_to_route_km": e["dist_route"],
                          "source": "pattern", "live_speed": None, "confidence": None,
                          "congestion_label": None, "dist_to_segment_m": None})

    _distribute_delay(spots, expected_delay_min)
    _normalize_alert_junctions(spots)        # identical, null-safe shape for the client

    # ── coverage + source breakdown (Component D) ──
    breakdown = {}
    for s in spots:
        breakdown[s["source"]] = breakdown.get(s["source"], 0) + 1
    mapbox_count  = breakdown.get("live_mapbox", 0)
    tomtom_count  = breakdown.get("live_tomtom", 0)
    pattern_count = breakdown.get("pattern", 0)
    nodata_count  = breakdown.get("none", 0)
    live_count    = sum(v for k, v in breakdown.items() if k.startswith("live_"))
    # primary_source = the source backing the most junctions this request
    buckets = {"mapbox_tiles": mapbox_count, "tomtom": tomtom_count,
               "patterns": pattern_count + nodata_count}
    primary = max(buckets, key=buckets.get) if spots else "patterns"

    pct  = round(100 * len(spots) / max(1, len(jids)))
    note = ("Spot levels are live (Mapbox traffic tiles, then TomTom, blended with "
            f"patterns) for junctions within {_LIVE_HORIZON_MIN} min, patterns beyond; "
            "the +min figure is the real Google route delay distributed across hotspots.")
    if not polyline_used:
        note += " Route geometry not provided — junction order/ETA is straight-line approximate."
    return spots, cov(len(spots), pct, polyline_used, note, extra={
        "live_junctions":    live_count,
        "pattern_junctions": pattern_count,
        "no_data_junctions": nodata_count,
        "primary_source":    primary,
        "tile_cache_hits":   budget["tile_cache_hits"],
        "source_breakdown":  breakdown,
        "mapbox_junctions":  mapbox_count,
        "tomtom_junctions":  tomtom_count,
        "tile_fetches":      budget["tile_used"],
        "tile_deferred":     budget["tile_deferred"],
        "tomtom_calls":      budget["tom_used"],
        "tomtom_deferred":   budget["tom_deferred"],
    })


async def _alert_junctions_from_scan(trip: dict, supabase, expected_delay_min: int):
    """
    Phase 3 read path. PREFER the per-junction model rows the scanner wrote in the
    latest scan batch (richer: live Mapbox + upstream feeder pressure, blended with
    patterns) over recomputing here. Returns (spots, coverage) or None if no such
    rows exist yet (→ caller falls back to the Phase 1+2 computation, untouched).

    Predictions (jam/level/source/feeders) are read directly; only the per-spot
    delay is re-distributed against the authoritative headline so the bubbles still
    sum EXACTLY to expected_delay_min (Phase 1 contract).
    """
    from services.scanner import _jam_to_level, _haversine_km
    trip_id = trip["id"]
    try:
        res = (supabase.table("scan_results").select("*")
               .eq("trip_id", trip_id).order("scanned_at", desc=True)
               .limit(120).execute())
    except Exception as e:
        logger.warning(f"scan_results read failed for {trip_id}: {e}")
        return None

    rows = res.data or []
    model_rows = [r for r in rows
                  if r.get("prediction_source") not in (None, "google_routes")
                  and r.get("predicted_level") != "ALL_CLEAR"]
    if not model_rows:
        return None

    latest_ts = max(r["scanned_at"] for r in model_rows)
    batch = [r for r in model_rows if r["scanned_at"] == latest_ts]
    if not batch:
        return None

    try:
        meta = ModelLoader.get_meta()
    except Exception:
        meta = None

    spots = []
    for r in batch:
        jid = r["junction_id"]
        uj  = r.get("upstream_junctions") or {}
        if not isinstance(uj, dict):
            uj = {}
        lat, lng = uj.get("lat"), uj.get("lng")
        if (lat is None or lng is None) and meta is not None:
            mrow = meta[meta["junction_id"] == jid]
            if not mrow.empty:
                lat = float(mrow.iloc[0]["latitude"]); lng = float(mrow.iloc[0]["longitude"])
        jam     = float(r.get("predicted_jam_factor") or 0.0)
        eta_sec = float(r.get("user_eta_seconds") or 0.0)
        spots.append({
            "junction_id": jid, "lat": lat, "lng": lng,
            "level": r.get("predicted_level") or _jam_to_level(jam),
            "jam_factor": round(jam, 2),
            "eta_minutes": round(eta_sec / 60, 1),
            "distance_from_origin_km": None,
            "distance_to_route_km": None,
            "source": r.get("prediction_source"),
            "live_speed": None, "confidence": None, "congestion_label": None,
            "feeder_pressure": uj.get("feeder_pressure"),
            "feeders": uj.get("feeders", []),
            "window": uj.get("window"),
            "lead_minutes": uj.get("lead_minutes"),
        })

    # order by ETA and fill distance_from_origin_km (rough, never null) so the
    # client gets the same non-null numeric the Phase 1+2 path always provided
    spots.sort(key=lambda s: s.get("eta_minutes") or 0.0)
    try:
        o_lat, o_lng = float(trip["origin_lat"]), float(trip["origin_lng"])
        d_lat, d_lng = float(trip["dest_lat"]),   float(trip["dest_lng"])
        total_km = _haversine_km(o_lat, o_lng, d_lat, d_lng)
        max_eta  = max((s.get("eta_minutes") or 0.0) for s in spots) or 1.0
        for s in spots:
            s["distance_from_origin_km"] = round((s.get("eta_minutes") or 0.0) / max_eta * total_km, 3)
            s["distance_to_route_km"]    = 0.0   # stored scan junctions are on-route
    except Exception:
        pass  # normalizer will default any remaining nulls

    # reconcile per-spot delay to the authoritative Google headline (Σ == expected)
    _distribute_delay(spots, expected_delay_min)
    _normalize_alert_junctions(spots)        # identical, null-safe shape for the client

    breakdown = {}
    for s in spots:
        breakdown[s["source"]] = breakdown.get(s["source"], 0) + 1
    live_count = sum(v for k, v in breakdown.items()
                     if k and (k.startswith("live_") or k == "feeder_blended"))
    coverage = {
        "trained_junctions_on_route": len(spots),
        "route_coverage_pct": round(100 * len(spots) / max(1, len(trip.get("junction_ids") or []))),
        "polyline_used": bool(trip.get("route_polyline")),
        "note": ("Per-junction levels are from the latest intelligent scan (Phase 3: "
                 "live Mapbox + upstream feeder pressure, blended with patterns). The "
                 "+min figure is the real Google route delay distributed across hotspots."),
        "live_junctions": live_count,
        "pattern_junctions": breakdown.get("pattern", 0),
        "no_data_junctions": breakdown.get("none", 0),
        "primary_source": "scan_results",
        "tile_cache_hits": 0, "tomtom_calls": 0,
        # legacy keys the Flutter client still expects — keep, never remove
        "tomtom_deferred": 0, "google_fallback_calls": 0,
        "source_breakdown": breakdown,
        "from_scan_results": True,
        "scan_batch_at": latest_ts,
    }
    return spots, coverage


@router.get("/trip-alert/{trip_id}")
async def get_trip_alert(trip_id: str):
    supabase = get_supabase()

    # ── 1. Trip data ──────────────────────────────────────────────
    try:
        trip_res = (
            supabase.table("planned_trips")
            .select("*")
            .eq("id", trip_id)
            .single()
            .execute()
        )
    except Exception as e:
        logger.error(f"Trip not found {trip_id}: {e}")
        raise HTTPException(status_code=404, detail="Trip not found")

    trip = trip_res.data
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    # ── 2. Latest scan alerts ─────────────────────────────────────
    try:
        scans_res = (
            supabase.table("scan_results")
            .select("*")
            .eq("trip_id", trip_id)
            .eq("alert_sent", True)
            .order("scanned_at", desc=True)
            .limit(10)
            .execute()
        )
    except Exception as e:
        logger.warning(f"Scans query failed for {trip_id}: {e}")
        scans_res = type("R", (), {"data": []})()

    # ── 3. Headline delay (magnitude anchor — Google live−free, unchanged) ──
    scans    = scans_res.data or []
    base_dur = float(trip.get("base_duration_seconds") or 0)
    live_sec = float(scans[0].get("user_eta_seconds") or 0) if scans else 0.0
    expected_delay_min = max(0, round((live_sec - base_dur) / 60)) if (scans and base_dur > 0) else 0
    route_seconds = live_sec if live_sec > 0 else base_dur

    # ── 4. Phase 3 preferred: per-junction rows from the latest intelligent scan.
    #      Fall back to Phase 1+2 live enrichment (untouched) if none exist yet. ──
    from_scan = await _alert_junctions_from_scan(trip, supabase, expected_delay_min)
    if from_scan is not None:
        alert_junctions, coverage = from_scan
    else:
        alert_junctions, coverage = await _build_per_spot_enrichment(
            trip, expected_delay_min, route_seconds)

    logger.info(
        f"Trip alert: {trip_id} | dest={trip.get('dest_name')} | "
        f"spots={len(alert_junctions)} | delay={expected_delay_min}min | "
        f"polyline={coverage.get('polyline_used')} | cov={coverage.get('route_coverage_pct')}% | "
        f"primary={coverage.get('primary_source')} | live={coverage.get('live_junctions')} | "
        f"from_scan={coverage.get('from_scan_results', False)} | "
        f"tiles(fetch={coverage.get('tile_fetches', 0)},cache={coverage.get('tile_cache_hits', 0)}) | "
        f"tomtom_calls={coverage.get('tomtom_calls', 0)} | sources={coverage.get('source_breakdown')}"
    )

    return {
        "trip":               trip,
        "alert_junctions":    alert_junctions,
        "expected_delay_min": expected_delay_min,
        "coverage":           coverage,
    }