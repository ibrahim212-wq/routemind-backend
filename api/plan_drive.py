"""
api/plan_drive.py
POST /api/plan-drive

The most important feature. Given an origin, destination, and target arrival time,
returns 76 time slots (every 15 min from 5:00 AM to 11:45 PM) with:
- Recommended departure time
- Predicted travel duration
- Traffic level

Uses Tier 1 (historical lookup) for all slots.
Uses Tier 2 (LSTM) only for slots within 2 hours from now.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from typing import Optional
import logging

from model.tier1 import predict_route_duration, jam_to_level
from services.junction_mapper import map_route_to_junctions

logger = logging.getLogger("routemind.plan_drive")
router = APIRouter()


# ── Request / Response ────────────────────────────────────────────────────────
class Waypoint(BaseModel):
    lat: float
    lng: float


class PlanDriveRequest(BaseModel):
    waypoints:             list[Waypoint]    # Route waypoints from Mapbox
    target_date:           str               # "2026-05-21"
    base_duration_seconds: int               # Free-flow travel time from Mapbox
    origin_lat:            float
    origin_lng:            float
    dest_lat:              float
    dest_lng:              float


class TimeSlot(BaseModel):
    slot_index:          int          # 0..75
    departure_time:      str          # "08:50"
    arrival_time_est:    str          # "10:02"
    duration_minutes:    float
    jam_factor:          float        # 0.0 - 1.0
    level:               str          # FREE / LIGHT / MODERATE / HIGH
    delay_minutes:       float        # extra minutes vs free-flow
    is_recommended:      bool
    confidence:          float


class PlanDriveResponse(BaseModel):
    slots:              list[TimeSlot]
    best_slot_index:    int
    best_departure:     str
    target_date:        str
    junctions_found:    int


# ── Core Logic ────────────────────────────────────────────────────────────────
def _time_from_slot(slot_idx: int) -> tuple[int, int]:
    """Slot 0 = 05:00, Slot 75 = 23:45. Step = 15 min."""
    total_min = 5 * 60 + slot_idx * 15
    return total_min // 60, total_min % 60


@router.post("/plan-drive", response_model=PlanDriveResponse)
async def plan_drive(req: PlanDriveRequest):
    """
    Returns 76 time slots for planning a drive.
    Flutter app displays these as scrollable cards.
    """
    target_date = date.fromisoformat(req.target_date)

    # Map waypoints to junction IDs
    waypoints_dicts = [{"lat": w.lat, "lng": w.lng} for w in req.waypoints]
    junctions = map_route_to_junctions(waypoints_dicts)
    junction_ids = [j["junction_id"] for j in junctions]

    logger.info(f"Plan Drive: {len(junction_ids)} junctions, date={req.target_date}")

    free_flow_minutes = req.base_duration_seconds / 60
    slots = []
    best_jam = float("inf")
    best_idx = 0

    for slot_idx in range(76):  # 5:00 AM → 11:45 PM
        hour, minute = _time_from_slot(slot_idx)
        departure_dt = datetime(
            target_date.year, target_date.month, target_date.day,
            hour, minute
        )

        # Tier 1 prediction
        pred = predict_route_duration(
            junction_ids,
            departure_dt,
            req.base_duration_seconds,
        )

        jam   = pred["jam_factor_avg"]
        dur_s = pred["predicted_seconds"]
        dur_m = dur_s / 60

        arrival_dt  = departure_dt + timedelta(seconds=dur_s)
        delay_min   = max(0.0, dur_m - free_flow_minutes)

        if jam < best_jam:
            best_jam = jam
            best_idx = slot_idx

        slots.append(TimeSlot(
            slot_index         = slot_idx,
            departure_time     = f"{hour:02d}:{minute:02d}",
            arrival_time_est   = f"{arrival_dt.hour:02d}:{arrival_dt.minute:02d}",
            duration_minutes   = round(dur_m, 1),
            jam_factor         = round(jam, 3),
            level              = pred["level"],
            delay_minutes      = round(delay_min, 1),
            is_recommended     = False,  # set after finding best
            confidence         = pred["confidence"],
        ))

    # Mark best slot
    if slots:
        slots[best_idx] = slots[best_idx].model_copy(update={"is_recommended": True})

    h, m = _time_from_slot(best_idx)

    return PlanDriveResponse(
        slots           = slots,
        best_slot_index = best_idx,
        best_departure  = f"{h:02d}:{m:02d}",
        target_date     = req.target_date,
        junctions_found = len(junction_ids),
    )