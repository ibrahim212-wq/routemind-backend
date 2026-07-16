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
# and the on-screen slots agree. The route-level jam_factor (0..1, formula
# (ratio-1)/0.50 vs a TRUE free-flow baseline — Google's predicted duration at
# 3:30 AM Cairo, see google_traffic.get_google_route_congestion) maps onto this
# 0..10 scale (×10):
#   slot LIGHT     jam01>=0.15 (ratio 1.075) → ADVISORY (1.5) — no alert
#   slot MODERATE  jam01>=0.32 (ratio 1.16)  → WARNING  (3.2) — alert
#   slot HIGH      jam01>=0.55 (ratio 1.275) → SERIOUS  (5.5) — alert
#   slot VERY_HIGH jam01>=0.80 (ratio 1.40)  → CRITICAL (8.0) — alert
# Empirically (Cairo, Thursday 16:10 rush, 2026-07-16, night baseline): Ring
# Road ratio 1.40-1.51, 6 Oct Bridge 1.50, Abbas El-Akkad 1.50, Salah Salem
# 1.20 — rush hour lands solidly in WARNING..CRITICAL, quiet hours in FREE.
JAM_LEVELS = [
    (8.0, "CRITICAL"),   # jam01 >= 0.80  (slot VERY_HIGH, ratio >= 1.40)
    (5.5, "SERIOUS"),    # jam01 >= 0.55  (slot HIGH,      ratio >= 1.275)
    (3.2, "WARNING"),    # jam01 >= 0.32  (slot MODERATE,  ratio >= 1.16)
    (1.5, "ADVISORY"),   # jam01 >= 0.15  (slot LIGHT,     ratio >= 1.075)
    (0.0, "FREE"),
]

LEVEL_ORDER = ["FREE", "ADVISORY", "WARNING", "SERIOUS", "CRITICAL"]

# Ratios alone overstate short trips (a 10-min trip at ratio 1.30 is +3 min —
# not worth waking anyone up). Absolute-delay floors demote the level until the
# delay in MINUTES actually justifies the word we're about to use.
_DELAY_FLOOR_CRITICAL = 18
_DELAY_FLOOR_SERIOUS  = 10
_DELAY_FLOOR_WARNING  = 5

# Pre-departure guarantee: every active trip gets exactly one send inside this
# window — a congestion reminder if the route is slow, ALL_CLEAR if it isn't.
# 20 min window + 10-min scan cadence ⇒ the send lands ~10 min before leave_by.
PRE_DEPARTURE_WINDOW_MINUTES = 20
ALL_CLEAR_WINDOW_MINUTES     = PRE_DEPARTURE_WINDOW_MINUTES  # back-compat alias

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

def _apply_delay_floors(level: str, delay_min: int) -> str:
    """Demote the level until the absolute delay justifies it (see floors)."""
    if level == "CRITICAL" and delay_min < _DELAY_FLOOR_CRITICAL:
        level = "SERIOUS"
    if level == "SERIOUS" and delay_min < _DELAY_FLOOR_SERIOUS:
        level = "WARNING"
    if level == "WARNING" and delay_min < _DELAY_FLOOR_WARNING:
        level = "ADVISORY"
    return level


async def _decide_alert(
    trip_id: str,
    level: str,
    delay_min: int,
    supabase: SupabaseClient,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Alert state machine. Compares this scan against the LAST congestion alert
    actually sent for the trip and returns (kind, prev_delay_min):

      "new"         — first congestion alert for this trip (level ≥ WARNING)
      "escalation"  — level rose, or the delay grew by ≥ 10 min at the same level
      "improvement" — was alerted congested, now clearly better (sent once)
      None          — nothing worth sending (unchanged / still clear)

    Reminder / ALL_CLEAR sends are excluded from the comparison (jsonb kind).
    """
    try:
        res = (supabase.table("scan_results")
               .select("alert_level,upstream_junctions,scanned_at")
               .eq("trip_id", trip_id)
               .eq("alert_sent", True)
               .eq("prediction_source", "google_routes")
               .order("scanned_at", desc=True)
               .limit(8)
               .execute())
        rows = [
            r for r in (res.data or [])
            if r.get("alert_level") not in (None, "ALL_CLEAR")
            and not (isinstance(r.get("upstream_junctions"), dict)
                     and r["upstream_junctions"].get("kind") == "reminder")
        ]
    except Exception as e:
        logger.warning(f"alert-decision read failed for {trip_id}: {e}")
        rows = []

    congested_now = level in ("WARNING", "SERIOUS", "CRITICAL")

    if not rows:
        return ("new", None) if congested_now else (None, None)

    last       = rows[0]
    last_level = last.get("alert_level") or "FREE"
    uj         = last.get("upstream_junctions")
    last_delay = uj.get("delay_min") if isinstance(uj, dict) else None

    li = LEVEL_ORDER.index(last_level) if last_level in LEVEL_ORDER else 0
    ci = LEVEL_ORDER.index(level)      if level in LEVEL_ORDER      else 0

    if congested_now and (
            ci > li
            or (last_delay is not None and delay_min >= int(last_delay) + 10)):
        return ("escalation", last_delay)

    # Was congested, now clearly better → one good-news send. The improvement
    # row itself becomes the new comparison point, so it never repeats.
    if li >= LEVEL_ORDER.index("WARNING") and (
            ci <= LEVEL_ORDER.index("ADVISORY")
            or ci <= li - 2
            or (last_delay is not None and delay_min <= int(last_delay) - 10)):
        return ("improvement", last_delay)

    return (None, last_delay)


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
    """Component F: distribute the real Google delay across the WORST 3
    congested junctions so per-junction delay_min sums EXACTLY to total_delay.
    Concentrating on 3 (instead of spreading over every hotspot) keeps the
    map chips meaningful — "+6 min" on the worst stretch instead of "+2 min"
    everywhere under a "+17 min" headline."""
    for s in spots:
        s["delay_min"] = 0
    if total_delay <= 0:
        return
    hot = sorted((s for s in spots
                  if s["level"] in ("WARNING", "SERIOUS", "CRITICAL")),
                 key=lambda s: s["jam10"], reverse=True)[:3]
    weighted = [(s, s["jam10"] / 10.0) for s in hot]
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


def _route_waypoints_for_congestion(trip: Dict, max_points: int = 8) -> List[Dict]:
    """Evenly sample the trip's stored route_polyline ([lat, lng] pairs) into
    via-waypoints so the Google congestion call is pinned to the USER'S route
    instead of whatever detour is currently fastest."""
    poly = trip.get("route_polyline") or []
    if not isinstance(poly, list) or len(poly) < 4:
        return []
    try:
        pts = [(float(p[0]), float(p[1])) for p in poly
               if isinstance(p, (list, tuple)) and len(p) >= 2]
    except (TypeError, ValueError):
        return []
    if len(pts) < 4:
        return []
    inner = pts[1:-1]
    step  = max(1, len(inner) // max_points)
    return [{"lat": la, "lng": ln} for la, ln in inner[::step]][:max_points]


def _reconcile_spot_levels(spots: List[Dict], route_jam10: float) -> None:
    """
    Re-center per-junction jams on the route-level anchor (upward only).

    WHY: Mapbox's congestion labels in Cairo are conservative — junctions read
    "low" even in real rush hour, so spot jams average ~1-2 while the pinned
    route-level measurement says 7-10. Without this the alert map renders a
    green route under a "+16 min" headline. The route anchor is the truth for
    MAGNITUDE; the live per-junction readings stay the truth for WHERE it is
    worse or better — so we shift every spot up to the anchor and AMPLIFY the
    relative differences (×3) so the map still shows which stretches are worst.
    """
    if not spots or not route_jam10:
        return
    target = min(float(route_jam10), 9.0)
    mean   = sum(s["jam10"] for s in spots) / len(spots)
    if target <= mean:
        return
    for s in spots:
        adj = max(0.0, min(10.0, target + (s["jam10"] - mean) * 3.0))
        s["jam10"] = round(adj, 2)
        s["level"] = _jam_to_level(adj)


def _worst_spot(spots: List[Dict]) -> Optional[Dict]:
    """The most congested analyzed junction (WARNING+), for copy like
    'the slow stretch hits ~12 min in'."""
    hot = [s for s in spots if s.get("level") in ("WARNING", "SERIOUS", "CRITICAL")]
    if not hot:
        return None
    return max(hot, key=lambda s: s.get("jam10", 0.0))


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

    # Derive junction_ids from the corridor if the trip was saved without any,
    # and WRITE THEM BACK into the in-memory trip — Phase 3 reads
    # trip["junction_ids"], and skipping this writeback used to silently kill
    # the whole per-junction analysis for safety-net trips (0 junctions).
    junction_ids = trip.get("junction_ids") or []
    if not junction_ids:
        junction_ids = _derive_junction_ids_for_trip(trip, supabase)
        trip["junction_ids"] = junction_ids
    rep_junction = junction_ids[0] if junction_ids else dest_name

    # ── Route congestion AT the departure time, pinned to the user's route,
    #    measured against a TRUE free-flow baseline (3:30 AM prediction) ──
    waypoints = _route_waypoints_for_congestion(trip)
    cong = await get_google_route_congestion(
        o_lat, o_lng, d_lat, d_lng,
        free_flow_seconds=trip.get("base_duration_seconds"),
        departure_utc=leave_by,
        waypoints=waypoints,
    )
    if cong is None:
        logger.error(
            f"Trip {trip_id} SKIPPED — Google congestion unavailable (dest={dest_name})"
        )
        return

    google_jam      = round(cong["jam_factor"] * 10, 2)   # 0..1 slot scale → 0..10
    scan_time       = now.isoformat()
    eta_minutes     = int(cong["live_seconds"] / 60)
    free_flow_sec   = cong["free_flow_seconds"]
    google_delay    = max(0, round((cong["live_seconds"] - free_flow_sec) / 60))

    logger.info(
        f"Trip {trip_id}: route congestion live={cong['live_seconds']}s "
        f"free={free_flow_sec}s ({cong.get('baseline_source')}) "
        f"ratio={cong['congestion_ratio']:.2f} | google_jam={google_jam:.1f} | "
        f"delay=+{google_delay}min | via={len(waypoints)}wp | "
        f"leave_by={leave_by.strftime('%H:%M')} | mins_to_depart={mins_to_departure:.0f}"
    )

    # ── Phase 3: per-junction spatial-propagation analysis (GCN graph feeders +
    #    live Mapbox tiles + Prophet/historical patterns at each junction's ETA) ──
    radj = reverse_adj if reverse_adj else _ensure_reverse_adj()
    spots: List[Dict] = []
    route_predicted_jam: Optional[float] = None
    try:
        spots, route_predicted_jam = await _analyze_route_junctions(
            trip, cong, radj, now, leave_by)
    except Exception as e:
        logger.warning(f"Trip {trip_id}: Phase 3 analysis failed ({e}) — Google-only fallback")
        spots, route_predicted_jam = [], None

    # ── Combine Google + model into the final severity ──
    # Google (route-pinned, true baseline) is the magnitude anchor. The model
    # layer can now RAISE severity when it has live evidence: up to +2.5 jam
    # (one level) above Google, capped at 6.5 (never model-only CRITICAL).
    # Model delay uses the same buffer semantics as the plan-drive slots
    # (duration multiplier = 1 + jam01·0.5).
    model_jam       = route_predicted_jam or 0.0
    has_live_model  = any(s.get("source") in ("live_mapbox", "feeder_blended")
                          for s in spots)
    model_delay_min = round(free_flow_sec * 0.5 * (model_jam / 10.0) / 60)

    final_jam = google_jam
    if model_jam >= _MODEL_FIRE_JAM and has_live_model and model_jam > google_jam:
        final_jam = min(model_jam, google_jam + 2.5, 6.5)

    real_delay_min = google_delay
    if final_jam > google_jam:
        real_delay_min = max(google_delay, model_delay_min)

    predicted_level = _apply_delay_floors(_jam_to_level(final_jam), real_delay_min)
    if predicted_level in ("WARNING", "SERIOUS", "CRITICAL"):
        _reconcile_spot_levels(spots, final_jam)               # anchor map to route truth
    _distribute_delay_scan(spots, real_delay_min)              # Component F

    logger.info(
        f"Trip {trip_id}: model_jam={route_predicted_jam} (live_evidence={has_live_model}, "
        f"model_delay=+{model_delay_min}min) google_jam={google_jam} → "
        f"final_jam={final_jam:.1f} {predicted_level} +{real_delay_min}min | "
        f"{len(spots)} junctions analyzed"
    )

    # ── Alert decision (new / escalation / improvement / none) ──
    kind, prev_delay = await _decide_alert(trip_id, predicted_level, real_delay_min, supabase)

    worst = _worst_spot(spots)
    await _store_scan_result({
        "trip_id":              trip_id,
        "junction_id":          rep_junction,
        "scanned_at":           scan_time,
        "user_eta_seconds":     cong["live_seconds"],
        "user_eta_datetime":    leave_by.isoformat(),
        "upstream_junctions":   {
            "delay_min":         real_delay_min,
            "free_flow_seconds": free_flow_sec,
            "google_jam":        google_jam,
            "model_jam":         route_predicted_jam,
            "baseline_source":   cong.get("baseline_source"),
            "kind":              kind or "scan",
            # minutes into the DRIVE (ETA from departure), not from now
            "worst_lead_minutes": (round(worst["eta_seconds"] / 60.0, 1)
                                   if worst and worst.get("eta_seconds") else None),
        },
        "predicted_jam_factor": round(final_jam, 2),
        "predicted_level":      predicted_level,
        "prediction_source":    "google_routes",
        "alert_sent":           kind is not None,
        "alert_level":          predicted_level if kind else None,
        "alert_sent_at":        scan_time if kind else None,
    }, supabase)

    # ── Component E: per-junction model rows (additive, alert_sent=False) ──
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

    # ── Congestion / improvement notification ──
    any_alert_sent = False
    if kind:
        any_alert_sent = True
        await _send_trip_alert_notification(
            user_id=user_id, trip_id=trip_id, dest_name=dest_name,
            junction_id=rep_junction, level=predicted_level, kind=kind,
            jam_factor=final_jam, eta_minutes=eta_minutes,
            delay_min=real_delay_min, prev_delay_min=prev_delay,
            leave_by=leave_by,
            worst_lead_min=(round(worst["eta_seconds"] / 60.0, 1)
                            if worst and worst.get("eta_seconds") else None),
            supabase=supabase,
        )

    # ── Pre-departure guarantee: exactly one send inside the window ──
    # Clear route → ALL_CLEAR. Congested route → "leaving soon, still slow"
    # reminder. If ANY alert (new/escalation/improvement) just went out this
    # same cycle, that send IS the pre-departure heads-up — don't double-send,
    # just mark the window as covered.
    final_clear_sent = bool(trip.get("final_clear_sent", False))
    if (not final_clear_sent
            and 0 <= mins_to_departure <= PRE_DEPARTURE_WINDOW_MINUTES):

        route_is_clear = predicted_level in ("FREE", "ADVISORY")

        if not any_alert_sent:
            await _store_scan_result({
                "trip_id":              trip_id,
                "junction_id":          rep_junction,
                "scanned_at":           scan_time,
                "user_eta_seconds":     cong["live_seconds"],
                "user_eta_datetime":    leave_by.isoformat(),
                "upstream_junctions":   {
                    "delay_min": real_delay_min,
                    "free_flow_seconds": free_flow_sec,
                    "kind": "all_clear" if route_is_clear else "reminder",
                },
                "predicted_jam_factor": round(final_jam, 2),
                "predicted_level":      "ALL_CLEAR" if route_is_clear else predicted_level,
                "prediction_source":    "google_routes",
                "alert_sent":           True,
                "alert_level":          "ALL_CLEAR" if route_is_clear else predicted_level,
                "alert_sent_at":        scan_time,
            }, supabase)

            if route_is_clear:
                await _send_all_clear_notification(
                    user_id=user_id, trip_id=trip_id, dest_name=dest_name,
                    junction_id=rep_junction, leave_by=leave_by,
                    supabase=supabase)
            else:
                await _send_departure_reminder_notification(
                    user_id=user_id, trip_id=trip_id, dest_name=dest_name,
                    junction_id=rep_junction, level=predicted_level,
                    delay_min=real_delay_min, leave_by=leave_by,
                    supabase=supabase)

        _mark_clear_sent(trip_id, scan_time, supabase)
        logger.info(
            f"Trip {trip_id}: pre-departure send "
            f"({'ALL_CLEAR' if route_is_clear else 'reminder' if not any_alert_sent else 'covered by alert'})"
        )

    logger.info(
        f"Trip {trip_id}: done | {predicted_level} (jam={final_jam:.1f}, "
        f"+{real_delay_min}min) | alert_kind={kind}"
    )


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


async def _send_fcm(
    user_id: str,
    title: str,
    body: str,
    data: Dict[str, str],
    supabase: SupabaseClient,
    log_tag: str,
) -> None:
    """Shared FCM plumbing: send to every registered token of the user."""
    tokens = _get_fcm_tokens(user_id, supabase)
    if not tokens:
        return

    from firebase_admin import messaging
    sent = 0
    for token in tokens:
        try:
            messaging.send(messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data,
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
            # dead/stale token — fine, the live ones still receive it
            logger.warning(f"{log_tag} send failed for one token of {user_id}: {e}")

    logger.info(f"{log_tag} → {user_id} | delivered to {sent}/{len(tokens)} token(s)")


def _leave_early_minutes(delay_min: int) -> int:
    """How much earlier to suggest leaving: delay + 20% cushion, rounded up to
    the nearest 5 (people plan in 5-minute chunks), minimum 5."""
    return max(5, int(math.ceil(delay_min * 1.2 / 5.0) * 5))


def _fmt_cairo_time(dt: datetime) -> str:
    local = dt.astimezone(CAIRO_TZ)
    return local.strftime("%I:%M %p").lstrip("0").lower()


def _short_dest(dest_name: str, limit: int = 30) -> str:
    d = (dest_name or "your destination").strip()
    # keep the part before the first comma ("Mall of Egypt, Wahat Rd, …")
    d = d.split(",")[0].strip() or d
    return d if len(d) <= limit else d[:limit - 1].rstrip() + "…"


def _notification_payload(trip_id: str, junction_id: str, level: str, kind: str,
                          jam_factor: float, eta_minutes: int, delay_min: int,
                          dest_name: str) -> Dict[str, str]:
    return {
        "trip_id":     trip_id,
        "junction_id": junction_id,
        "level":       level,
        "kind":        kind,
        "jam_factor":  str(jam_factor),
        "eta_minutes": str(eta_minutes),
        "delay_min":   str(delay_min),
        "dest_name":   dest_name or "",
        "tap_action":  "show_route_details",
    }


async def _send_trip_alert_notification(
    user_id: str,
    trip_id: str,
    dest_name: str,
    junction_id: str,
    level: str,
    kind: str,                       # "new" | "escalation" | "improvement"
    jam_factor: float,
    eta_minutes: int,
    delay_min: int,
    prev_delay_min: Optional[int],
    leave_by: datetime,
    worst_lead_min: Optional[float],
    supabase: SupabaseClient,
) -> None:
    dest  = _short_dest(dest_name)
    early = _leave_early_minutes(delay_min)
    lead  = int(worst_lead_min) if worst_lead_min and worst_lead_min >= 3 else None
    spot  = f" — the slow stretch hits about {lead} min into your drive" if lead else ""

    if kind == "improvement":
        if delay_min < _DELAY_FLOOR_WARNING or level in ("FREE", "ADVISORY"):
            title = f"Road's clearing up for {dest} ✨"
            body  = ("That earlier traffic has melted away. You're good to "
                     "leave right on schedule — enjoy the drive.")
        else:
            title = f"Good news — traffic to {dest} is easing"
            body  = (f"It's down to about {delay_min} extra minutes now"
                     + (f" (was +{prev_delay_min})" if prev_delay_min else "")
                     + ". Your planned time should work again.")
    elif kind == "escalation":
        title = f"Traffic to {dest} just got heavier"
        body  = (f"The slowdown grew since we last checked — now about "
                 f"{delay_min} extra minutes"
                 + (f", up from +{prev_delay_min}" if prev_delay_min else "")
                 + f". If you can head out {early} min early, now's the moment.")
    elif level == "CRITICAL":
        title = f"Heavy traffic on the way to {dest} 🚦"
        body  = (f"It's rough out there — about {delay_min} extra minutes on "
                 f"your route{spot}. Leave {early} min early and you'll still "
                 f"make it comfortably.")
    elif level == "SERIOUS":
        title = f"Traffic is building toward {dest}"
        body  = (f"Your route is running about {delay_min} minutes slower than "
                 f"usual{spot}. A {early}-minute head start keeps you on time.")
    else:  # WARNING
        title = f"Slight slowdown on the {dest} route"
        body  = (f"A bit of traffic{spot} — roughly {delay_min} extra minutes. "
                 f"Nothing major, but leave a few minutes early if you're "
                 f"cutting it close.")

    await _send_fcm(
        user_id, title, body,
        _notification_payload(trip_id, junction_id, level, kind,
                              jam_factor, eta_minutes, delay_min, dest_name),
        supabase, log_tag=f"Alert[{kind}/{level}] trip={trip_id}",
    )


async def _send_all_clear_notification(
    user_id: str,
    trip_id: str,
    dest_name: str,
    junction_id: str,
    leave_by: datetime,
    supabase: SupabaseClient,
) -> None:
    dest = _short_dest(dest_name)
    t    = _fmt_cairo_time(leave_by)
    title = f"You're all set for {dest} ✨"
    body  = (f"Just gave your route a final check — all clear. Leave at {t} "
             f"as planned and enjoy a smooth drive.")

    await _send_fcm(
        user_id, title, body,
        _notification_payload(trip_id, junction_id, "ALL_CLEAR", "all_clear",
                              0.0, 0, 0, dest_name),
        supabase, log_tag=f"AllClear trip={trip_id}",
    )


async def _send_departure_reminder_notification(
    user_id: str,
    trip_id: str,
    dest_name: str,
    junction_id: str,
    level: str,
    delay_min: int,
    leave_by: datetime,
    supabase: SupabaseClient,
) -> None:
    dest = _short_dest(dest_name)
    t    = _fmt_cairo_time(leave_by)
    title = f"Leaving soon? {dest} still has traffic"
    body  = (f"You planned to head out at {t} — the route is still carrying "
             f"about {delay_min} extra minutes. Leaving right now beats "
             f"waiting it out.")

    await _send_fcm(
        user_id, title, body,
        _notification_payload(trip_id, junction_id, level, "reminder",
                              0.0, 0, delay_min, dest_name),
        supabase, log_tag=f"Reminder[{level}] trip={trip_id}",
    )


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