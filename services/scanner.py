"""
services/scanner.py

Intelligent Scan engine — يربط كل الخطوات مع بعض.

التعديلات:
  - إضافة trip_id لـ _send_notification signature
  - إضافة trip_id لـ FCM data payload
  - تمرير trip_id في الـ call من scan_trip
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

logger = logging.getLogger("routemind.scanner")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

JAM_LEVELS = [
    (8.0, "CRITICAL"),
    (6.0, "SERIOUS"),
    (4.0, "WARNING"),
    (2.0, "ADVISORY"),
    (0.0, "FREE"),
]

MIN_ALERT_INTERVAL_MINUTES = 15

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

    hour    = eta_datetime.hour
    dow     = eta_datetime.weekday()
    slot_15 = (hour * 60 + eta_datetime.minute) // 15

    jam = hist.get((junction_id, dow, slot_15))
    if jam is None:
        for offset in [1, -1, 2, -2, 3, -3]:
            jam = hist.get((junction_id, dow, slot_15 + offset))
            if jam is not None:
                break

    if jam is None:
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

        feats   = prophet[junction_id]
        slot_15 = (eta_datetime.hour * 60 + eta_datetime.minute) // 15
        dow     = eta_datetime.weekday()

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
    if level == "FREE":
        return False

    cutoff = (datetime.now(timezone.utc)
              - timedelta(minutes=MIN_ALERT_INTERVAL_MINUTES)).isoformat()

    try:
        res = (supabase.table("scan_results")
               .select("alert_level")
               .eq("trip_id", trip_id)
               .eq("junction_id", junction_id)
               .eq("alert_sent", True)
               .gte("scanned_at", cutoff)
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

async def scan_trip(
    trip: Dict,
    reverse_adj: Dict,
    supabase: SupabaseClient,
) -> None:
    trip_id      = trip["id"]
    user_id      = trip["user_id"]
    junction_ids = trip.get("junction_ids") or []
    base_dur     = float(trip.get("base_duration_seconds", 3600))
    leave_by     = datetime.fromisoformat(trip["leave_by"])
    now          = datetime.now(timezone.utc)
    combined_meta = ModelLoader.get_combined_meta()

    if not junction_ids:
        return

    etas = _estimate_etas(junction_ids, base_dur, leave_by)
    if not etas:
        return

    logger.info(
        f"Trip {trip_id}: {len(etas)} junctions | "
        f"leave_by={leave_by.strftime('%H:%M')} | "
        f"duration={base_dur/60:.0f}min"
    )

    readings_batch = []
    scan_time      = now.isoformat()

    for eta_info in etas:
        junction_id  = eta_info["junction_id"]
        eta_seconds  = eta_info["eta_seconds"]
        eta_datetime = eta_info["eta_datetime"]
        jlat         = eta_info["latitude"]
        jlon         = eta_info["longitude"]

        upstream_junctions = get_upstream_junctions(
            target_junction_id=junction_id,
            time_offset_seconds=eta_seconds,
            meta_df=combined_meta,
            reverse_adj=reverse_adj,
            tolerance_pct=0.25,
        )

        junctions_to_scan = upstream_junctions if upstream_junctions else [{
            "junction_id": junction_id,
            "latitude":    jlat,
            "longitude":   jlon,
        }]

        upstream_readings = []
        for uj in junctions_to_scan[:5]:
            flow = await _fetch_mapbox_flow(uj["latitude"], uj["longitude"])
            if flow is None:
                continue

            readings_batch.append({
                "junction_id":      uj["junction_id"],
                "recorded_at":      scan_time,
                "jam_factor":       flow["jam_factor"],
                "current_speed":    flow["current_speed"],
                "free_flow_speed":  flow["free_flow_speed"],
                "congestion_ratio": flow["congestion_ratio"],
                "speed_reduction":  flow["speed_reduction"],
                "confidence":       flow["confidence"],
                "hour":             now.hour,
                "day_of_week":      now.weekday(),
                "is_weekend":       now.weekday() >= 5,
                "is_friday":        now.weekday() == 4,
                "scan_trigger":     "intelligent_scan",
            })

            upstream_readings.append({
                "junction_id":     uj["junction_id"],
                "travel_time_sec": uj.get("travel_time_to_target_sec", 0),
                "jam_factor":      flow["jam_factor"],
                "current_speed":   flow["current_speed"],
            })

        historical = _get_historical_prediction(junction_id, eta_datetime)
        prophet    = _get_prophet_prediction(junction_id, eta_datetime)
        ema        = _get_ema_prediction(junction_id, supabase)

        predicted_jam, predicted_level = _ensemble(
            historical, prophet, ema, upstream_readings
        )

        sources = []
        if historical:        sources.append("historical")
        if prophet:           sources.append("prophet")
        if ema:               sources.append("ema")
        if upstream_readings: sources.append("mapbox_upstream")
        prediction_source = "+".join(sources) if sources else "mapbox_only"

        should_alert = await _should_send_alert(
            trip_id, junction_id, predicted_level, supabase
        )

        await _store_scan_result({
            "trip_id":              trip_id,
            "junction_id":          junction_id,
            "scanned_at":           scan_time,
            "user_eta_seconds":     eta_seconds,
            "user_eta_datetime":    eta_datetime.isoformat(),
            "upstream_junctions":   upstream_readings,
            "predicted_jam_factor": predicted_jam,
            "predicted_level":      predicted_level,
            "prediction_source":    prediction_source,
            "alert_sent":           should_alert,
            "alert_level":          predicted_level if should_alert else None,
            "alert_sent_at":        scan_time if should_alert else None,
        }, supabase)

        # ── Notification ─────────────────────────────────────────
        if should_alert:
            await _send_notification(
                user_id=user_id,
                trip_id=trip_id,
                junction_id=junction_id,
                level=predicted_level,
                jam_factor=predicted_jam,
                eta_minutes=int(eta_seconds / 60),
                supabase=supabase,
            )

        logger.info(
            f"  {junction_id}: ETA={eta_seconds/60:.0f}min | "
            f"jam={predicted_jam:.1f} | {predicted_level} | "
            f"src={prediction_source} | alert={should_alert}"
        )

    await _store_readings(readings_batch, supabase)
    logger.info(f"Trip {trip_id}: done | {len(readings_batch)} readings stored")


# ─────────────────────────────────────────────────────────────────
# Notification
# ─────────────────────────────────────────────────────────────────

async def _send_notification(
    user_id: str,
    trip_id: str,
    junction_id: str,
    level: str,
    jam_factor: float,
    eta_minutes: int,
    supabase: SupabaseClient,
) -> None:
    messages = {
        "CRITICAL": f"⛔ زحمة شديدة جداً بعد {eta_minutes} دقيقة",
        "SERIOUS":  f"🔴 زحمة كبيرة متوقعة بعد {eta_minutes} دقيقة",
        "WARNING":  f"🟡 زحمة متوسطة بعد {eta_minutes} دقيقة",
        "ADVISORY": f"🟢 تباطؤ بسيط متوقع بعد {eta_minutes} دقيقة",
    }
    body = messages.get(level, "")
    if not body:
        return

    try:
        token_res = (supabase.table("fcm_tokens")
                     .select("token")
                     .eq("user_id", user_id)
                     .limit(1)
                     .execute())
        if not token_res.data:
            return
        fcm_token = token_res.data[0]["token"]

        from firebase_admin import messaging
        messaging.send(messaging.Message(
            notification=messaging.Notification(
                title="RouteMind — تنبيه مرور",
                body=body,
            ),
            data={
                "trip_id":     trip_id,
                "junction_id": junction_id,
                "level":       level,
                "jam_factor":  str(jam_factor),
                "eta_minutes": str(eta_minutes),
            },
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="routemind_trips",
                    click_action="FLUTTER_NOTIFICATION_CLICK",
                ),
            ),
            token=fcm_token,
        ))
        logger.info(f"Notification → {user_id}: {level} @ {junction_id} (trip={trip_id})")

    except Exception as e:
        logger.error(f"Failed to send notification to {user_id}: {e}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

_reverse_adj_cache = None


async def run_intelligent_scan(supabase: SupabaseClient) -> None:
    global _reverse_adj_cache

    if _reverse_adj_cache is None:
        edge_index, _ = ModelLoader.get_graph()
        combined_meta = ModelLoader.get_combined_meta()
        _reverse_adj_cache = build_reverse_adj(edge_index, combined_meta)
        logger.info(f"Reverse adjacency built: {len(combined_meta)} nodes")

    now     = datetime.now(timezone.utc)
    horizon = (now + timedelta(minutes=90)).isoformat()

    try:
        res = (supabase.table("planned_trips")
               .select("*")
               .eq("status", "active")
               .lte("leave_by", horizon)
               .gte("leave_by", now.isoformat())
               .execute())
    except Exception as e:
        logger.error(f"Failed to fetch trips: {e}")
        return

    trips = res.data or []
    if not trips:
        logger.info("Intelligent Scan: مفيش trips نشطة")
        return

    logger.info(f"Intelligent Scan: {len(trips)} trip(s)")

    for trip in trips:
        try:
            await scan_trip(trip, _reverse_adj_cache, supabase)
        except Exception as e:
            logger.error(f"scan_trip failed {trip.get('id')}: {e}")

    logger.info("Intelligent Scan: done")