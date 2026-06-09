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

import logging
logger = logging.getLogger("routemind.trip_alert")

router = APIRouter()


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

    # ── 3. Junction coordinates من ModelLoader ────────────────────
    meta = ModelLoader.get_combined_meta()
    seen_junctions = set()
    alert_junctions = []

    for scan in (scans_res.data or []):
        jid = scan.get("junction_id")
        if not jid or jid in seen_junctions:
            continue
        seen_junctions.add(jid)

        row = meta[meta["junction_id"] == jid]
        if row.empty:
            continue

        alert_junctions.append({
            "junction_id":  jid,
            "lat":          float(row.iloc[0]["latitude"]),
            "lng":          float(row.iloc[0]["longitude"]),
            "level":        scan.get("predicted_level", "ADVISORY"),
            "jam_factor":   float(scan.get("predicted_jam_factor", 0.0)),
            "eta_minutes":  round(float(scan.get("user_eta_seconds", 0)) / 60, 1),
            "scanned_at":   scan.get("scanned_at"),
        })

    logger.info(
        f"Trip alert: {trip_id} | "
        f"dest={trip.get('dest_name')} | "
        f"junctions={len(alert_junctions)}"
    )

    return {
        "trip":             trip,
        "alert_junctions":  alert_junctions,
    }