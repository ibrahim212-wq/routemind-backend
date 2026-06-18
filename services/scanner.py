"""
services/scanner.py

Intelligent Scan engine — يربط كل الخطوات مع بعض.

النظام الموحّد:
  - بيقيس الزحمة المتوقعة على كل junction
  - لو فيه زحمة → notification "leave early"
  - لو الطريق فاضي + قرّب الميعاد → ALL_CLEAR مرة واحدة
  - بيخزّن كل النتايج في scan_results (تظهر في Alert History)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import httpx
from services.supabase_client import SupabaseClient, get_supabase
from model.loader import ModelLoader
from services.upstream_discovery import get_upstream_junctions, build_reverse_adj
from services.google_traffic import get_google_route_congestion
from services.junction_eta import compute_junction_etas

logger = logging.getLogger("routemind.scanner")

# Africa/Cairo local time. Historical + Prophet seasonal patterns are keyed by
# LOCAL time-of-day (hour / 15-min slot), but trip timestamps are stored and
# handled in UTC. We must convert before indexing those lookups.
try:
    from zoneinfo import ZoneInfo
    CAIRO_TZ = ZoneInfo("Africa/Cairo")
except Exception:  # pragma: no cover — missing tzdata; fall back to fixed UTC+2
    CAIRO_TZ = timezone(timedelta(hours=2))
    logger.warning("zoneinfo 'Africa/Cairo' unavailable — falling back to fixed UTC+2")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

# Levels are aligned 1:1 with the plan-drive slot scale so that scanner alerts
# and the on-screen slots agree. The route-level Google jam_factor (0..1, the
# SAME signal/formula the slots use) is mapped onto this 0..10 scale (×10):
#   slot LIGHT    jam01>=0.15  → ADVISORY  (1.5)   — no alert
#   slot MODERATE jam01>=0.32  → WARNING   (3.2)   — alert
#   slot HIGH     jam01>=0.55  → SERIOUS   (5.5)   — alert
#   slot VERY_HIGH jam01>=0.80 → CRITICAL  (8.0)   — alert
# Empirically (Cairo, 2026-06-16): Salah Salem jam01≈0.17→ADVISORY (quiet, no
# alert), 6 Oct Bridge ≈0.44→WARNING, Ring Road ≈0.64→SERIOUS. The old
# micro-segment speed-ratio signal maxed jam01≈0.10 city-wide, so it could
# never cross WARNING — that was cause (B).
JAM_LEVELS = [
    (8.0, "CRITICAL"),   # jam01 >= 0.80  (slot VERY_HIGH)
    (5.5, "SERIOUS"),    # jam01 >= 0.55  (slot HIGH)
    (2.5, "WARNING"),    # jam01 >= 0.25  (ratio >= ~1.125) — catches moderate Cairo rush-hour
    (1.5, "ADVISORY"),   # jam01 >= 0.15  (slot LIGHT)
    (0.0, "FREE"),
]

MIN_ALERT_INTERVAL_MINUTES = 15

# لو باقي على leave_by أقل من ده وكله فاضي → ابعت ALL_CLEAR
# 20 دقيقة + scan كل 10 دقايق = مضمون إن scan واحد على الأقل يقع في الـ window
# فالـ user بياخد "اتحرك في معادك" حوالي 10 دقايق قبل القيام
ALL_CLEAR_WINDOW_MINUTES = 20

# ─────────────────────────────────────────────────────────────────
# Geo + Level helpers
# ─────────────────────────────────────────────────────────────────

def _jam_to_level(jam: float) -> str:
    for threshold, level in JAM_LEVELS:
        if jam >= threshold:
            return level
    return "FREE"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2-lat1)/2)**2
         + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2)
    return 2 * R * math.asin(math.sqrt(max(0, min(1, a))))


# ─────────────────────────────────────────────────────────────────
# ETA Estimation
# ─────────────────────────────────────────────────────────────────

def _estimate_etas(
    junction_ids: List[str],
    base_duration_seconds: float,
    departure_time: datetime,
) -> List[Dict]:
    meta_df = ModelLoader.get_combined_meta()
    junctions = []

    for jid in junction_ids:
        row = meta_df[meta_df["junction_id"] == jid]
        if row.empty:
            continue
        r = row.iloc[0]
        junctions.append({
            "junction_id":  jid,
            "junction_idx": int(r["junction_idx"]),
            "latitude":     float(r["latitude"]),
            "longitude":    float(r["longitude"]),
        })

    if len(junctions) < 2:
        if len(junctions) == 1:
            eta_sec = base_duration_seconds / 2
            return [{**junctions[0],
                     "distance_from_origin_km": 0.0,
                     "eta_seconds":  eta_sec,
                     "eta_datetime": departure_time + timedelta(seconds=eta_sec)}]
        return []

    cumulative = [0.0]
    for i in range(1, len(junctions)):
        d = _haversine_km(
            junctions[i-1]["latitude"], junctions[i-1]["longitude"],
            junctions[i]["latitude"],   junctions[i]["longitude"],
        )
        cumulative.append(cumulative[-1] + d)

    total_dist = cumulative[-1]
    if total_dist <= 0:
        return []

    results = []
    for i, j in enumerate(junctions):
        eta_sec = (cumulative[i] / total_dist) * base_duration_seconds
        results.append({
            **j,
            "distance_from_origin_km": round(cumulative[i], 3),
            "eta_seconds":  round(eta_sec, 1),
            "eta_datetime": departure_time + timedelta(seconds=eta_sec),
        })
    return results


# ─────────────────────────────────────────────────────────────────
# Mapbox Flow
# ─────────────────────────────────────────────────────────────────

async def _fetch_mapbox_flow(lat: float, lon: float) -> Optional[Dict]:
    from services.mapbox_traffic import get_mapbox_segment_speed
    offset = 0.001
    try:
        return await get_mapbox_segment_speed(
            lat - offset, lon,
            lat + offset, lon,
        )
    except Exception as e:
        logger.warning(f"Mapbox flow exception ({lat},{lon}): {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Prediction Pipeline
# ─────────────────────────────────────────────────────────────────

def _get_historical_prediction(
    junction_id: str,
    eta_datetime: datetime,
) -> Optional[Tuple[float, float]]:
    hist = ModelLoader.get_hist_lookup()
    if hist is None:
        return None

    # historical_lookup.pkl is keyed by (junction_id, hour[0-23], dow[0-6])
    # — verified against the artifact (460×24×7 = 77,280 dense entries).
    # eta_datetime is UTC-aware; convert to Cairo local before indexing.
    local = eta_datetime.astimezone(CAIRO_TZ)
    hour  = local.hour
    dow   = local.weekday()

    jam = hist.get((junction_id, hour, dow))
    if jam is None:
        # Real junctions form a dense grid, so a miss means this id is not in
        # the historical set (e.g. a virtual junction). Surface it instead of
        # silently dropping the 0.35-weight historical signal.
        logger.debug(
            f"historical lookup miss: {junction_id} @ Cairo hour={hour} dow={dow}"
        )
        return None

    return round(float(jam) * 10, 2), 0.75


def _get_prophet_prediction(
    junction_id: str,
    eta_datetime: datetime,
) -> Optional[Tuple[float, float]]:
    try:
        prophet = ModelLoader.get_prophet_feats()
        if prophet is None or junction_id not in prophet:
            return None

        # Prophet's daily curve is indexed by 15-min slot of the LOCAL day and
        # weekly by local day-of-week — convert from UTC to Cairo to match.
        feats   = prophet[junction_id]
        local   = eta_datetime.astimezone(CAIRO_TZ)
        slot_15 = (local.hour * 60 + local.minute) // 15
        dow     = local.weekday()

        daily_val  = float(feats["daily"][slot_15])
        weekly_val = float(feats["weekly"][dow])

        combined = daily_val * 0.7 + weekly_val * 0.3
        return round(combined * 10, 2), 0.70

    except Exception:
        return None


def _get_ema_prediction(
    junction_id: str,
    supabase: SupabaseClient,
) -> Optional[Tuple[float, float]]:
    try:
        res = (supabase.table("junction_readings")
               .select("jam_factor")
               .eq("junction_id", junction_id)
               .order("recorded_at", desc=True)
               .limit(12)
               .execute())

        readings = res.data or []
        if not readings:
            return None

        jams = [float(r["jam_factor"]) for r in readings]

        if len(jams) >= 12:   confidence = 0.80
        elif len(jams) >= 6:  confidence = 0.65
        else:                 confidence = 0.45

        alpha = 0.3
        ema   = jams[0]
        for j in jams[1:]:
            ema = alpha * j + (1 - alpha) * ema

        return round(ema, 2), confidence

    except Exception:
        return None


def _ensemble(
    historical: Optional[Tuple[float, float]],
    prophet:    Optional[Tuple[float, float]],
    ema:        Optional[Tuple[float, float]],
    upstream_readings: List[Dict],
) -> Tuple[float, str]:
    available = {}
    if historical: available["historical"] = historical
    if prophet:    available["prophet"]    = prophet
    if ema:        available["ema"]        = ema

    if not available:
        if upstream_readings:
            avg = sum(r["jam_factor"] for r in upstream_readings) / len(upstream_readings)
            return round(avg, 2), _jam_to_level(avg)
        return 0.0, "FREE"

    weights = {"historical": 0.35, "prophet": 0.30, "ema": 0.35}
    total_weight = sum(weights[k] for k in available)
    base_jam = sum(
        (weights[k] / total_weight) * available[k][0]
        for k in available
    )

    if upstream_readings:
        avg_upstream = sum(r["jam_factor"] for r in upstream_readings) / len(upstream_readings)
        final_jam = max(0.0, round(0.60 * base_jam + 0.40 * avg_upstream, 2))
    else:
        final_jam = max(0.0, round(base_jam, 2))

    return final_jam, _jam_to_level(final_jam)


# ─────────────────────────────────────────────────────────────────
# Anti-spam + Storage
# ─────────────────────────────────────────────────────────────────

async def _should_send_alert(
    trip_id: str,
    junction_id: str,
    level: str,
    supabase: SupabaseClient,
) -> bool:
    # FREE + ADVISORY = طريق سالك — مش بنبعت congestion alert
    # (قبل كده ADVISORY كان بيبعت alert وده كان بيمنع الـ ALL_CLEAR
    #  من إنه يتبعت رغم إن route_is_clear بيعتبر ADVISORY فاضي — تناقض)
    if level in ("FREE", "ADVISORY"):
        return False

    # Escalation-only dedup (mirrors the final_clear_sent "send once" intent):
    # resend ONLY if this level is strictly higher than the highest level already
    # alerted for the trip. Same-or-lower = no resend, with NO time window — the
    # old 15-min cutoff was the spam source (#4): once it expired a still-congested
    # trip re-alerted every cycle. Latest recorded alert == highest, because lower
    # levels are suppressed and never get an alert_sent row.
    try:
        res = (supabase.table("scan_results")
               .select("alert_level")
               .eq("trip_id", trip_id)
               .eq("junction_id", junction_id)
               .eq("alert_sent", True)
               .order("scanned_at", desc=True)
               .limit(1)
               .execute())

        if res.data:
            last_level  = res.data[0]["alert_level"]
            level_order = ["FREE", "ADVISORY", "WARNING", "SERIOUS", "CRITICAL"]
            if (last_level in level_order and level in level_order
                    and level_order.index(level) <= level_order.index(last_level)):
                return False
    except Exception as e:
        logger.warning(f"Anti-spam check failed: {e}")

    return True


async def _store_readings(readings: List[Dict], supabase: SupabaseClient) -> None:
    if not readings:
        return
    try:
        supabase.table("junction_readings").insert(readings)
    except Exception as e:
        logger.error(f"Failed to store {len(readings)} readings: {e}")


async def _store_scan_result(result: Dict, supabase: SupabaseClient) -> None:
    try:
        supabase.table("scan_results").insert(result)
    except Exception as e:
        logger.error(f"Failed to store scan result: {e}")


# ─────────────────────────────────────────────────────────────────
# Main: scan trip واحدة
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# junction_ids safety-net
# ─────────────────────────────────────────────────────────────────

def _sample_line_waypoints(
    o_lat: float, o_lng: float, d_lat: float, d_lng: float,
    step_km: float = 1.5,
) -> List[Dict]:
    """Evenly-spaced points along the straight origin→dest line."""
    total = _haversine_km(o_lat, o_lng, d_lat, d_lng)
    n = max(2, int(total / step_km) + 1)
    return [
        {"lat": o_lat + (d_lat - o_lat) * (i / n),
         "lng": o_lng + (d_lng - o_lng) * (i / n)}
        for i in range(n + 1)
    ]


def _derive_junction_ids_for_trip(trip: Dict, supabase: SupabaseClient) -> List[str]:
    """
    Safety net for trips persisted with empty junction_ids (e.g. the mobile app
    saved the trip before the plan-drive stream emitted its `done` event, or the
    stream failed). Re-maps the origin→dest corridor to real junctions and
    persists them back so the work happens only once.

    Uses a straight-line corridor (no external HTTP) — coarser than the actual
    route, but the scan also pulls live upstream traffic, and *some* junctions
    is vastly better than skipping the trip entirely.
    """
    try:
        o_lat = float(trip["origin_lat"]); o_lng = float(trip["origin_lng"])
        d_lat = float(trip["dest_lat"]);   d_lng = float(trip["dest_lng"])
    except (KeyError, TypeError, ValueError):
        return []

    from services.junction_mapper import map_route_to_junctions
    waypoints = _sample_line_waypoints(o_lat, o_lng, d_lat, d_lng)
    ids = [j["junction_id"] for j in map_route_to_junctions(waypoints)]
    if not ids:
        return []

    try:
        (supabase.table("planned_trips")
         .update({"junction_ids": ids})
         .eq("id", trip["id"])
         .execute_update())
    except Exception as e:
        logger.warning(
            f"Could not persist re-derived junction_ids for {trip.get('id')}: {e}"
        )

    return ids


# ─────────────────────────────────────────────────────────────────
# Phase 3 — spatial-propagation analysis (feeder/upstream awareness)
# Additive: enriches WHERE/severity + per-junction storage. The firing
# decision stays Google-gated (_should_send_alert) — model never fires a
# false alert. Every step degrades gracefully to Google-only on failure.
# ─────────────────────────────────────────────────────────────────

_LIVE_WINDOW_MIN  = 60      # junction reached within this → live window, else patterns
_FEEDER_K         = 5       # max feeders per junction (cost control)
_FEEDER_HOPS      = 2       # max upstream hops for feeders (cost control)
_MODEL_FIRE_JAM   = 3.2     # WARNING threshold on the 0..10 scale
_SCAN_TILE_BUDGET = 60      # max Mapbox tile fetches per trip scan (shared)


def _ensure_reverse_adj() -> Dict:
    """Lazily build (and cache) the reverse adjacency. Used by scan-now, which
    passes an empty reverse_adj. Returns {} on failure (feeders then no-op)."""
    global _reverse_adj_cache
    if _reverse_adj_cache is None:
        try:
            edge_index, _ = ModelLoader.get_graph()
            combined_meta = ModelLoader.get_combined_meta()
            _reverse_adj_cache = build_reverse_adj(edge_index, combined_meta)
            logger.info(f"Reverse adjacency built lazily: {len(_reverse_adj_cache)} nodes")
        except Exception as e:
            logger.warning(f"reverse_adj build failed: {e} — feeder scan disabled")
            _reverse_adj_cache = {}
    return _reverse_adj_cache


async def _live_jam10(lat: float, lon: float, client, budget) -> Optional[float]:
    """get_live_congestion → jam10 (0..10) or None (no_data / budget / error)."""
    try:
        from services.mapbox_traffic import get_live_congestion
        r = await get_live_congestion(lat, lon, client=client, budget=budget)
        if r and r.get("jam10") is not None:
            return float(r["jam10"])
    except Exception as e:
        logger.debug(f"live_jam10 ({lat},{lon}) failed: {e}")
    return None


def _blend_live_scan(live_jam10: float, jid: str, arrival) -> float:
    """Component C (live window): 0.5·live + 0.25·prophet + 0.25·historical,
    all on the 0..10 scale. Weights renormalized if a pattern is missing."""
    parts = [(0.5, max(0.0, min(10.0, live_jam10)))]
    h = _get_historical_prediction(jid, arrival)
    if h: parts.append((0.25, max(0.0, min(10.0, h[0]))))
    p = _get_prophet_prediction(jid, arrival)
    if p: parts.append((0.25, max(0.0, min(10.0, p[0]))))
    wsum = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / wsum if wsum > 0 else 0.0


def _pattern_ensemble(jid: str, arrival) -> float:
    """Component C (pattern window): patterns-only ensemble (ema=None)."""
    hist  = _get_historical_prediction(jid, arrival)
    proph = _get_prophet_prediction(jid, arrival)
    jam, _level = _ensemble(hist, proph, None, [])
    return jam


def _distribute_delay_scan(spots: List[Dict], total_delay: int) -> None:
    """Component F: distribute the real Google delay across predicted-congested
    junctions so per-junction delay_min sums EXACTLY to total_delay (same logic
    as Phase 1 _distribute_delay; replicated here to avoid a trip_alert import)."""
    for s in spots:
        s["delay_min"] = 0
    if total_delay <= 0:
        return
    weighted = [(s, s["jam10"] / 10.0) for s in spots
                if s["level"] in ("WARNING", "SERIOUS", "CRITICAL")]
    wsum = sum(w for _, w in weighted)
    if wsum <= 0:
        if spots:
            max(spots, key=lambda s: s["jam10"])["delay_min"] = total_delay
        return
    raw = [(s, total_delay * w / wsum) for s, w in weighted]
    for s, r in raw:
        s["delay_min"] = int(r)
    remainder = total_delay - sum(s["delay_min"] for s, _ in raw)
    for s, _ in sorted(raw, key=lambda x: x[1], reverse=True):
        if remainder <= 0:
            break
        s["delay_min"] += 1
        remainder -= 1


async def _analyze_route_junctions(
    trip: Dict, cong: Dict, reverse_adj: Dict,
    now: datetime, leave_by: datetime,
) -> Tuple[List[Dict], Optional[float]]:
    """
    Components A-C. For each on-route junction: classify live (≤60min) vs pattern
    window; in the live window fetch direct live congestion + feeder pressure from
    upstream roads; ensemble with patterns; map to level. Returns (spots,
    route_predicted_jam). Never raises — returns ([], None) on failure.
    """
    import httpx

    try:
        meta460  = ModelLoader.get_meta()
        combined = ModelLoader.get_combined_meta()
    except Exception as e:
        logger.warning(f"Phase3: meta unavailable ({e})")
        return [], None

    jids = trip.get("junction_ids") or []
    ordered, seen = [], set()
    for jid in jids:
        if jid not in seen:
            seen.add(jid); ordered.append(jid)
    if not ordered:
        return [], None

    # in-model (trained 460) lookup
    in_model: Dict[str, Tuple[float, float, int]] = {}
    for jid in ordered:
        row = meta460[meta460["junction_id"] == jid]
        if not row.empty:
            r = row.iloc[0]
            in_model[jid] = (float(r["latitude"]), float(r["longitude"]), int(r["junction_idx"]))

    base_dur = float(trip.get("base_duration_seconds") or 0)
    dur      = float(cong.get("live_seconds") or base_dur or 1800.0)
    polyline = trip.get("route_polyline") or []

    def _coords(jid):
        if jid in in_model:
            return in_model[jid][0], in_model[jid][1]
        row = combined[combined["junction_id"] == jid]
        if not row.empty:
            r = row.iloc[0]
            return float(r["latitude"]), float(r["longitude"])
        return None

    # ── ETA list: polyline preferred, straight-line fallback ──
    eta_rows: List[Dict] = []
    if isinstance(polyline, list) and len(polyline) >= 2:
        try:
            pl  = [(float(p[0]), float(p[1])) for p in polyline]
            jor = []
            for jid in ordered:
                c = _coords(jid)
                if c:
                    jor.append({"junction_id": jid,
                                "junction_idx": in_model.get(jid, (0, 0, -1))[2],
                                "latitude": c[0], "longitude": c[1]})
            eta_rows = compute_junction_etas(pl, jor, dur, leave_by)
        except Exception as ex:
            logger.warning(f"Phase3 polyline ETA failed ({ex}) — straight-line")
            eta_rows = []
    if not eta_rows:
        eta_rows = _estimate_etas(ordered, dur, leave_by)
    if not eta_rows:
        return [], None

    budget = {"tile_used": 0, "tile_max": _SCAN_TILE_BUDGET,
              "tile_cache_hits": 0, "tile_deferred": 0}
    spots: List[Dict] = []

    async with httpx.AsyncClient(timeout=6.0) as client:
        for e in eta_rows:
            jid     = e["junction_id"]
            lat     = float(e["latitude"]); lon = float(e["longitude"])
            arrival = e["eta_datetime"]
            eta_sec = float(e["eta_seconds"])
            lead_min = max(0.0, (arrival - now).total_seconds() / 60.0)
            is_in_model = jid in in_model

            direct_jam = feeder_pressure = None
            feeders_used: List[Dict] = []
            source = "pattern"

            if lead_min <= _LIVE_WINDOW_MIN:
                # Component A — live window: direct live congestion
                direct_jam = await _live_jam10(lat, lon, client, budget)

                # Component B — feeder/upstream pressure (in-model only).
                # Spatial propagation along the actual road graph (reverse-Dijkstra):
                # find feeders whose travel-time TO this junction ≈ lead (±25%) —
                # roads whose traffic is moving NOW and will arrive at this junction
                # exactly when the USER does. Skip when the user is almost here
                # (<5 min): those feeders are already reflected in the direct live
                # reading. No hop cap — a 40-min-away feeder is many hops upstream.
                if is_in_model and reverse_adj and lead_min >= 5.0:
                    lead_sec = lead_min * 60.0
                    try:
                        ups = get_upstream_junctions(
                            jid, lead_sec, combined, reverse_adj,
                            tolerance_pct=0.25, max_hops=None)
                    except Exception as ex:
                        logger.warning(f"Phase3 upstream {jid} failed: {ex}")
                        ups = []
                    ups = sorted(ups, key=lambda f: f["delta_sec"])[:_FEEDER_K]
                    fvals = []
                    for f in ups:
                        fj = await _live_jam10(f["latitude"], f["longitude"], client, budget)
                        if fj is None:            # skip no_data feeders (B)
                            continue
                        tt    = f["travel_time_to_target_sec"]
                        delta = f["delta_sec"]
                        w     = 1.0 / (delta + 1.0)   # closest to the ETA match → heavier
                        fvals.append((fj, w))
                        feeders_used.append({"junction_id": f["junction_id"],
                                             "jam10": round(fj, 2),
                                             "travel_time_sec": tt,
                                             "delta_sec": delta})
                    if fvals:
                        feeder_pressure = round(
                            sum(j * w for j, w in fvals) / sum(w for _, w in fvals), 2)

                # blend direct + feeder (B)
                if direct_jam is not None and feeder_pressure is not None:
                    live_blend = 0.6 * direct_jam + 0.4 * feeder_pressure
                    source = "feeder_blended"
                elif direct_jam is not None:
                    live_blend = direct_jam
                    source = "live_mapbox"
                elif feeder_pressure is not None:
                    live_blend = feeder_pressure
                    source = "feeder_blended"
                else:
                    live_blend = None

                if live_blend is not None:
                    jam10 = _blend_live_scan(live_blend, jid, arrival)   # Component C
                elif is_in_model:
                    jam10 = _pattern_ensemble(jid, arrival); source = "pattern"
                else:
                    continue   # out-of-model + no live → skip (Component C)
            else:
                # Component A/C — pattern window
                if is_in_model:
                    jam10 = _pattern_ensemble(jid, arrival); source = "pattern"
                else:
                    continue   # out-of-model in pattern window → skip

            jam10 = round(max(0.0, min(10.0, jam10)), 2)
            spots.append({
                "junction_id": jid, "lat": lat, "lng": lon,
                "jam10": jam10, "level": _jam_to_level(jam10),
                "source": source,
                "window": "live" if lead_min <= _LIVE_WINDOW_MIN else "pattern",
                "lead_minutes": round(lead_min, 1),
                "eta_seconds": round(eta_sec, 1),
                "eta_datetime": arrival.isoformat() if hasattr(arrival, "isoformat") else str(arrival),
                "direct_live_jam10": round(direct_jam, 2) if direct_jam is not None else None,
                "feeder_pressure": feeder_pressure,
                "feeders": feeders_used,
                "in_model": is_in_model,
                "delay_min": 0,
            })

    if not spots:
        return [], None

    # Component D — route_predicted_jam: weight nearer-to-departure junctions more
    wsum = sum(1.0 / (s["lead_minutes"] + 1.0) for s in spots)
    route_jam = (round(sum(s["jam10"] * (1.0 / (s["lead_minutes"] + 1.0))
                           for s in spots) / wsum, 2) if wsum > 0 else 0.0)
    logger.info(
        f"Phase3 budget: tiles fetched={budget['tile_used']} "
        f"cache={budget['tile_cache_hits']} deferred={budget['tile_deferred']}"
    )
    return spots, route_jam


async def scan_trip(
    trip: Dict,
    reverse_adj: Dict,          # built by run_intelligent_scan; empty from scan-now (lazy-built)
    supabase: SupabaseClient,
) -> None:
    trip_id   = trip["id"]
    user_id   = trip["user_id"]
    dest_name = trip.get("dest_name", "your destination")
    leave_by  = _parse_dt(trip["leave_by"])
    now       = datetime.now(timezone.utc)
    mins_to_departure = (leave_by - now).total_seconds() / 60.0

    # Origin / destination are required for the route-level congestion lookup.
    try:
        o_lat = float(trip["origin_lat"]); o_lng = float(trip["origin_lng"])
        d_lat = float(trip["dest_lat"]);   d_lng = float(trip["dest_lng"])
    except (KeyError, TypeError, ValueError):
        logger.error(
            f"Trip {trip_id} SKIPPED — missing origin/dest coordinates (dest={dest_name})"
        )
        return

    # scan_results.junction_id wants a real junction id (anti-spam keys on it).
    # Derive from the corridor if the trip was saved without any.
    junction_ids = trip.get("junction_ids") or []
    if not junction_ids:
        junction_ids = _derive_junction_ids_for_trip(trip, supabase)
    rep_junction = junction_ids[0] if junction_ids else dest_name

    base_dur = float(trip.get("base_duration_seconds") or 0)

    # Predicted congestion AT the user's departure time, from the SAME Google
    # Routes source + jam formula the plan-drive slots use → alerts match slots.
    # free_flow_seconds=None → use Google's own TRAFFIC_UNAWARE free-flow as the
    # baseline (the stored base_duration_seconds was often inflated by congestion at
    # trip-creation, masking real congestion: ratio<1 → jam=0 → no alerts).
    cong = await get_google_route_congestion(
        o_lat, o_lng, d_lat, d_lng,
        free_flow_seconds=None,
        departure_utc=leave_by,
    )
    if cong is None:
        logger.error(
            f"Trip {trip_id} SKIPPED — Google congestion unavailable (dest={dest_name})"
        )
        return

    predicted_jam   = round(cong["jam_factor"] * 10, 2)   # 0..1 slot scale → 0..10 level scale
    predicted_level = _jam_to_level(predicted_jam)
    scan_time       = now.isoformat()
    eta_minutes     = int(cong["live_seconds"] / 60)
    # The one honest delay: extra minutes vs free-flow (live − free), never negative.
    real_delay_min  = max(0, round((cong["live_seconds"] - cong["free_flow_seconds"]) / 60))

    logger.info(
        f"Trip {trip_id}: route congestion live={cong['live_seconds']}s "
        f"free={cong['free_flow_seconds']}s ratio={cong['congestion_ratio']:.2f} | "
        f"jam={predicted_jam:.1f} | {predicted_level} | "
        f"leave_by={leave_by.strftime('%H:%M')} | mins_to_depart={mins_to_departure:.0f}"
    )

    # ── Phase 3: per-junction spatial-propagation analysis (Components A-C, F) ──
    # Purely additive: enriches WHERE/severity and per-junction storage. The
    # firing gate below stays Google-driven, so this can NEVER cause a false alert.
    radj = reverse_adj if reverse_adj else _ensure_reverse_adj()
    spots: List[Dict] = []
    route_predicted_jam: Optional[float] = None
    try:
        spots, route_predicted_jam = await _analyze_route_junctions(
            trip, cong, radj, now, leave_by)
        _distribute_delay_scan(spots, real_delay_min)          # Component F
    except Exception as e:
        logger.warning(f"Trip {trip_id}: Phase 3 analysis failed ({e}) — Google-only fallback")
        spots, route_predicted_jam = [], None

    # ── Component D: confidence classification (LOGGING ONLY — Google stays the
    #    firing gate; the model never fires on its own, never suppresses a real
    #    Google alert) ──
    google_congested = predicted_level in ("WARNING", "SERIOUS", "CRITICAL")
    model_congested  = (route_predicted_jam is not None
                        and route_predicted_jam >= _MODEL_FIRE_JAM)
    if   model_congested and google_congested:     confidence = "HIGH (model+Google agree)"
    elif model_congested and not google_congested: confidence = "LOW (model only → suppressed, trust Google)"
    elif google_congested and not model_congested: confidence = "ROUTE (Google only → route-level alert)"
    else:                                          confidence = "CLEAR (both clear)"
    logger.info(
        f"Trip {trip_id}: Phase3 model_jam={route_predicted_jam} google_jam={predicted_jam} | "
        f"{len(spots)} junctions analyzed | confidence={confidence}"
    )

    should_alert = await _should_send_alert(trip_id, rep_junction, predicted_level, supabase)

    await _store_scan_result({
        "trip_id":              trip_id,
        "junction_id":          rep_junction,
        "scanned_at":           scan_time,
        "user_eta_seconds":     cong["live_seconds"],
        "user_eta_datetime":    leave_by.isoformat(),
        "upstream_junctions":   [],
        "predicted_jam_factor": predicted_jam,
        "predicted_level":      predicted_level,
        "prediction_source":    "google_routes",
        "alert_sent":           should_alert,
        "alert_level":          predicted_level if should_alert else None,
        "alert_sent_at":        scan_time if should_alert else None,
    }, supabase)

    # ── Component E: per-junction model rows (additive, alert_sent=False) ──
    # Same scanned_at as the google_routes rep row above. The rich extras
    # (lat/lng/feeder_pressure/lead/delay/feeders) ride in the upstream_junctions
    # jsonb column (no dedicated columns exist). trip_alert reads these rows
    # (prediction_source != "google_routes"). Failure never breaks the scan.
    if spots:
        try:
            rows = [{
                "trip_id":              trip_id,
                "junction_id":          s["junction_id"],
                "scanned_at":           scan_time,
                "user_eta_seconds":     s["eta_seconds"],
                "user_eta_datetime":    s["eta_datetime"],
                "upstream_junctions":   {
                    "lat":               s["lat"],
                    "lng":               s["lng"],
                    "window":            s["window"],
                    "lead_minutes":      s["lead_minutes"],
                    "direct_live_jam10": s["direct_live_jam10"],
                    "feeder_pressure":   s["feeder_pressure"],
                    "feeders":           s["feeders"],
                    "delay_min":         s["delay_min"],
                    "in_model":          s["in_model"],
                },
                "predicted_jam_factor": s["jam10"],
                "predicted_level":      s["level"],
                "prediction_source":    s["source"],
                "alert_sent":           False,
                "alert_level":          None,
                "alert_sent_at":        None,
            } for s in spots]
            supabase.table("scan_results").insert(rows)
            logger.info(f"Trip {trip_id}: stored {len(rows)} per-junction rows")
        except Exception as e:
            logger.warning(f"Trip {trip_id}: per-junction storage failed ({e})")

    # ── Congestion notification ──────────────────────────────────
    any_alert_sent = False
    if should_alert:
        any_alert_sent = True
        await _send_congestion_notification(
            user_id=user_id,
            trip_id=trip_id,
            dest_name=dest_name,
            junction_id=rep_junction,
            level=predicted_level,
            jam_factor=predicted_jam,
            eta_minutes=eta_minutes,
            real_delay_min=real_delay_min,
            supabase=supabase,
        )

    # ── ALL_CLEAR logic ──────────────────────────────────────────
    # لو الطريق سالك + قرّب الميعاد + لسه مبعتناش all_clear
    final_clear_sent = bool(trip.get("final_clear_sent", False))
    route_is_clear   = (predicted_level in ("FREE", "ADVISORY")) and not any_alert_sent

    if (route_is_clear
            and not final_clear_sent
            and 0 <= mins_to_departure <= ALL_CLEAR_WINDOW_MINUTES):

        await _store_scan_result({
            "trip_id":              trip_id,
            "junction_id":          rep_junction,
            "scanned_at":           scan_time,
            "user_eta_seconds":     cong["live_seconds"],
            "user_eta_datetime":    leave_by.isoformat(),
            "upstream_junctions":   [],
            "predicted_jam_factor": predicted_jam,
            "predicted_level":      "ALL_CLEAR",
            "prediction_source":    "google_routes",
            "alert_sent":           True,
            "alert_level":          "ALL_CLEAR",
            "alert_sent_at":        scan_time,
        }, supabase)

        await _send_all_clear_notification(
            user_id=user_id,
            trip_id=trip_id,
            dest_name=dest_name,
            junction_id=rep_junction,
            leave_by=leave_by,
            supabase=supabase,
        )

        _mark_clear_sent(trip_id, scan_time, supabase)
        logger.info(f"Trip {trip_id}: ALL_CLEAR sent")

    logger.info(f"Trip {trip_id}: done | {predicted_level} (jam={predicted_jam:.1f})")


# ─────────────────────────────────────────────────────────────────
# Trip flag update
# ─────────────────────────────────────────────────────────────────

def _mark_clear_sent(trip_id: str, scan_time: str, supabase: SupabaseClient) -> None:
    try:
        (supabase.table("planned_trips")
         .update({"final_clear_sent": True, "alert_sent_at": scan_time})
         .eq("id", trip_id)
         .execute_update())
    except Exception as e:
        logger.error(f"Failed to mark clear_sent for {trip_id}: {e}")


def _parse_dt(s: str) -> datetime:
    """Parse ISO timestamp, ensure UTC-aware."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ─────────────────────────────────────────────────────────────────
# Notifications (English)
# ─────────────────────────────────────────────────────────────────

def _get_fcm_tokens(user_id: str, supabase: SupabaseClient) -> List[str]:
    """
    بيرجع كل الـ tokens بتاعة الـ user (مش واحد عشوائي).
    قبل كده كان limit(1) بدون ترتيب — مع وجود tokens قديمة في الجدول
    كان ممكن يبعت لـ token ميت والـ notification تضيع بصمت.
    دلوقتي بنبعت لكل الـ tokens — الحي هيستقبل والميت هيفشل لوحده.
    """
    try:
        res = (supabase.table("fcm_tokens")
               .select("token")
               .eq("user_id", user_id)
               .limit(10)
               .execute())
        tokens = [r["token"] for r in (res.data or []) if r.get("token")]
        if not tokens:
            logger.error(
                f"NO FCM TOKEN for user {user_id} — "
                f"notification CANNOT be delivered. "
                f"(الـ app لازم يحفظ الـ token في fcm_tokens عند التشغيل)"
            )
        return tokens
    except Exception as e:
        logger.error(f"Failed to get FCM tokens for {user_id}: {e}")
        return []


async def _send_congestion_notification(
    user_id: str,
    trip_id: str,
    dest_name: str,
    junction_id: str,
    level: str,
    jam_factor: float,
    eta_minutes: int,
    real_delay_min: int,
    supabase: SupabaseClient,
) -> None:
    titles = {
        "CRITICAL": "Heavy traffic ahead — leave early",
        "SERIOUS":  "Traffic building on your route",
        "WARNING":  "Slowdown expected on your route",
        "ADVISORY": "Light traffic on your route",
    }
    # Copy uses the honest delay (extra minutes vs free-flow), not the full
    # trip duration. eta_minutes is intentionally NOT shown as "into your trip".
    bodies = {
        "CRITICAL": f"Heavy traffic to {dest_name} — expect about +{real_delay_min} min. "
                    f"Leave early to stay on time. Tap to see affected roads.",
        "SERIOUS":  f"Traffic building on your route to {dest_name} — about +{real_delay_min} min added. "
                    f"Consider leaving a bit earlier. Tap for live conditions.",
        "WARNING":  f"Some traffic to {dest_name} — about +{real_delay_min} min added. "
                    f"Tap to view details.",
        "ADVISORY": f"A minor slowdown is expected on your route to {dest_name}. "
                    f"Roads are mostly clear.",
    }
    title = titles.get(level)
    body  = bodies.get(level)
    if not title or not body:
        return

    tokens = _get_fcm_tokens(user_id, supabase)
    if not tokens:
        return

    from firebase_admin import messaging
    sent = 0
    for token in tokens:
        try:
            messaging.send(messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data={
                    "trip_id":     trip_id,
                    "junction_id": junction_id,
                    "level":       level,
                    "jam_factor":  str(jam_factor),
                    "eta_minutes": str(eta_minutes),
                    "tap_action":  "show_route_details",
                },
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        channel_id="routemind_trips",
                        click_action="FLUTTER_NOTIFICATION_CLICK",
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(title=title, body=body),
                            sound="default",
                        ),
                    ),
                ),
                token=token,
            ))
            sent += 1
        except Exception as e:
            # token ميت/قديم — عادي، باقي الـ tokens هتوصل
            logger.warning(f"Congestion send failed for one token of {user_id}: {e}")

    logger.info(
        f"Congestion → {user_id}: {level} @ {junction_id} "
        f"(trip={trip_id}) | delivered to {sent}/{len(tokens)} token(s)"
    )


async def _send_all_clear_notification(
    user_id: str,
    trip_id: str,
    dest_name: str,
    junction_id: str,
    leave_by: datetime,
    supabase: SupabaseClient,
) -> None:
    title = "All clear — you're good to go"
    body  = (f"No significant traffic detected on your route to {dest_name}. "
             f"Leave as planned and have a safe trip!")

    tokens = _get_fcm_tokens(user_id, supabase)
    if not tokens:
        return

    from firebase_admin import messaging
    sent = 0
    for token in tokens:
        try:
            messaging.send(messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data={
                    "trip_id":     trip_id,
                    "junction_id": junction_id,
                    "level":       "ALL_CLEAR",
                    "eta_minutes": "0",
                    "tap_action":  "show_route_details",
                },
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        channel_id="routemind_trips",
                        click_action="FLUTTER_NOTIFICATION_CLICK",
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(title=title, body=body),
                            sound="default",
                        ),
                    ),
                ),
                token=token,
            ))
            sent += 1
        except Exception as e:
            logger.warning(f"ALL_CLEAR send failed for one token of {user_id}: {e}")

    logger.info(f"ALL_CLEAR → {user_id} (trip={trip_id}) | delivered to {sent}/{len(tokens)} token(s)")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

_reverse_adj_cache = None


async def run_intelligent_scan(supabase: SupabaseClient) -> None:
    # Build (once) the reverse adjacency used by Phase 3 feeder discovery.
    reverse_adj = _ensure_reverse_adj()

    now     = datetime.now(timezone.utc)
    horizon = (now + timedelta(minutes=90)).isoformat()
    # 5-min grace: a trip whose leave_by just passed still gets one final scan
    # instead of being silently dropped between cycles.
    grace   = (now - timedelta(minutes=5)).isoformat()

    try:
        res = (supabase.table("planned_trips")
               .select("*")
               .eq("status", "active")
               .lte("leave_by", horizon)
               .gte("leave_by", grace)
               .execute())
    except Exception as e:
        logger.error(f"Failed to fetch trips: {e}")
        return

    trips = res.data or []
    if not trips:
        logger.info("Intelligent Scan: no active trips")
        return

    logger.info(f"Intelligent Scan: {len(trips)} trip(s)")

    for trip in trips:
        try:
            await scan_trip(trip, reverse_adj, supabase)
        except Exception as e:
            logger.error(f"scan_trip failed {trip.get('id')}: {e}")

    logger.info("Intelligent Scan: done")