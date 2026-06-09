"""
api/plan_drive.py
POST /api/plan-drive

Returns 76 time slots (every 15 min, 5:00 AM → 11:45 PM) with predicted
travel duration, traffic level, and recommended departure time.

Source selection:
  coverage ≥ 70%  →  our trained historical model  (confidence 0.85)
  coverage < 70%  →  TomTom Route API fallback      (confidence 0.75)
    - 29 real calls at strategic departure times
    - remaining slots filled by linear interpolation (confidence 0.65)
  TomTom failure  →  model as last resort            (confidence 0.50)

TomTom note:
  TomTom only has accurate historical traffic for PAST dates.
  For future target dates we use the most recent past date with the
  same day-of-week, so Thursday predictions use last Thursday's data,
  Friday predictions use last Friday's data, etc.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta, date
import logging

from model.tier1 import predict_route_duration, jam_to_level
from services.junction_mapper import (
    map_route_to_junctions,
    calculate_route_coverage,
    COVERAGE_MIN_PCT,
)
from services.tomtom import get_tomtom_plan

logger = logging.getLogger("routemind.plan_drive")
router = APIRouter()

# Confidence levels
CONF_MODEL        = 0.85
CONF_TOMTOM       = 0.75
CONF_INTERPOLATED = 0.65
CONF_FALLBACK     = 0.50


# ── Request / Response ────────────────────────────────────────────────────────

class Waypoint(BaseModel):
    lat: float
    lng: float


class PlanDriveRequest(BaseModel):
    waypoints:             list[Waypoint]   # route coords sampled from Mapbox geometry
    target_date:           str              # "2026-05-21"
    base_duration_seconds: int              # FREE-FLOW travel time from Mapbox driving profile
    origin_lat:            float
    origin_lng:            float
    dest_lat:              float
    dest_lng:              float


class TimeSlot(BaseModel):
    slot_index:       int
    departure_time:   str     # "08:30"
    arrival_time_est: str     # "09:47"
    duration_minutes: float
    jam_factor:       float   # 0.0–1.0
    level:            str     # FREE / LIGHT / MODERATE / HIGH
    delay_minutes:    float
    is_recommended:   bool
    confidence:       float


class PlanDriveResponse(BaseModel):
    slots:              list[TimeSlot]
    best_slot_index:    int
    best_departure:     str
    target_date:        str
    junctions_found:    int
    junction_ids:       list[str]       # ← جديد
    route_coverage:     float   # 0.0–1.0
    source:             str     # "model" | "tomtom" | "fallback"
    tomtom_reference_date: str  # actual date used for TomTom calls


# ── Helpers ───────────────────────────────────────────────────────────────────

def _time_from_slot(slot_idx: int) -> tuple[int, int]:
    """Slot 0 = 05:00, Slot 75 = 23:45, step = 15 min."""
    total = 5 * 60 + slot_idx * 15
    return total // 60, total % 60


def _jam_from_ratio(duration_s: int, base_s: int) -> float:
    """
    Back-calculates jam_factor (0–1) from TomTom duration vs free-flow.
    Inverse of jam_to_multiplier: multiplier = 1 + jam * 1.5
    """
    ratio = duration_s / max(base_s, 1)
    return round(max(0.0, min(1.0, (ratio - 1.0) / 1.5)), 3)


def _tomtom_date(target: date) -> date:
    """
    Returns the best date to send to TomTom for traffic predictions.

    TomTom's predictive model (future dates) gives better variation for
    Egyptian roads than recorded historical data (past dates), because
    Egypt's road coverage in TomTom's historical database is limited.

    Rules:
      - target is today or future -> use target as-is
        (TomTom uses its statistical/predictive model -> good variation)
      - target is in the past -> use next occurrence of same weekday
        (keeps the day-of-week pattern: Thu->next Thu, Fri->next Fri)
    """
    today = date.today()
    if target >= today:
        return target
    days_ahead = (target.weekday() - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def _interpolate_slots(
    tomtom_data: list[tuple[int, int, int]],
    base_s: int,
) -> dict[int, tuple[int, float]]:
    """
    Builds duration + confidence for all 76 slots from TomTom anchor points
    using linear interpolation between anchors.

    Returns:
        {slot_index: (duration_seconds, confidence)}
    """
    anchors    = sorted((h * 60 + m, d) for h, m, d in tomtom_data)
    anchor_min = {h * 60 + m for h, m, _ in tomtom_data}

    result: dict[int, tuple[int, float]] = {}

    for idx in range(76):
        slot_min = 5 * 60 + idx * 15

        # Exact match → full TomTom confidence
        if slot_min in anchor_min:
            dur = next(d for t, d in anchors if t == slot_min)
            result[idx] = (dur, CONF_TOMTOM)
            continue

        # Find surrounding anchors for interpolation
        before = next(((t, d) for t, d in reversed(anchors) if t < slot_min), None)
        after  = next(((t, d) for t, d in anchors if t > slot_min), None)

        if before and after:
            ratio = (slot_min - before[0]) / (after[0] - before[0])
            dur   = int(before[1] + ratio * (after[1] - before[1]))
            result[idx] = (dur, CONF_INTERPOLATED)
        elif before:
            result[idx] = (before[1], CONF_INTERPOLATED)
        elif after:
            result[idx] = (after[1], CONF_INTERPOLATED)
        else:
            result[idx] = (base_s, CONF_FALLBACK)

    return result


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/plan-drive", response_model=PlanDriveResponse)
async def plan_drive(req: PlanDriveRequest):
    """
    Returns 76 departure-time slots for planning a drive.

    Decision logic:
      1. Sample the route waypoints from Mapbox geometry
      2. Calculate coverage: % of waypoints with a training-data junction within 2.5km
      3. coverage ≥ 70% → our model (fast, free, accurate for Cairo/Giza inner roads)
         coverage < 70% → TomTom using last week's same weekday for real historical data
      4. If TomTom also fails → model as last resort
    """
    target_date = date.fromisoformat(req.target_date)

    # ── 1. Build waypoints list ───────────────────────────────────────────────
    waypoints_dicts = [{"lat": w.lat, "lng": w.lng} for w in req.waypoints]
    if not waypoints_dicts:
        waypoints_dicts = [
            {"lat": req.origin_lat, "lng": req.origin_lng},
            {"lat": req.dest_lat,   "lng": req.dest_lng},
        ]

    # ── 2. Coverage check ─────────────────────────────────────────────────────
    coverage     = calculate_route_coverage(waypoints_dicts)
    junctions    = map_route_to_junctions(waypoints_dicts)
    junction_ids = [j["junction_id"] for j in junctions]

    logger.info(
        f"Plan Drive: coverage={coverage:.0%}, junctions={len(junction_ids)}, "
        f"target={req.target_date}, origin=({req.origin_lat:.4f},{req.origin_lng:.4f})"
    )

    # ── 3. Decide source ──────────────────────────────────────────────────────
    use_tomtom = coverage < COVERAGE_MIN_PCT

    tomtom_slots: dict[int, tuple[int, float]] | None = None
    source         = "model"
    reference_date = req.target_date   # default: same as target

    if use_tomtom:
        past_date      = _tomtom_date(target_date)
        reference_date = past_date.isoformat()

        logger.info(
            f"Plan Drive: coverage {coverage:.0%} < {COVERAGE_MIN_PCT:.0%} "
            f"→ TomTom with reference date {reference_date} "
            f"(same weekday as {req.target_date})"
        )

        tomtom_data = await get_tomtom_plan(
            req.origin_lat, req.origin_lng,
            req.dest_lat,   req.dest_lng,
            past_date,
        )

        if tomtom_data:
            tomtom_slots = _interpolate_slots(tomtom_data, req.base_duration_seconds)
            source = "tomtom"
            logger.info(f"Plan Drive: TomTom returned {len(tomtom_data)}/29 slots")
        else:
            logger.warning("Plan Drive: TomTom failed → falling back to model")
            source = "fallback"

    # ── 4. Build 76 slots ─────────────────────────────────────────────────────
    free_flow_minutes = req.base_duration_seconds / 60
    slots: list[TimeSlot] = []
    best_dur = float("inf")
    best_idx = 0

    for idx in range(76):
        hour, minute = _time_from_slot(idx)
        departure_dt = datetime(
            target_date.year, target_date.month, target_date.day, hour, minute
        )

        if tomtom_slots:
            dur_s, confidence = tomtom_slots[idx]
            dur_m     = dur_s / 60
            jam       = _jam_from_ratio(dur_s, req.base_duration_seconds)
            level     = jam_to_level(jam)
            delay_min = max(0.0, dur_m - free_flow_minutes)
        else:
            pred       = predict_route_duration(
                junction_ids, departure_dt, req.base_duration_seconds
            )
            dur_s      = pred["predicted_seconds"]
            dur_m      = dur_s / 60
            jam        = pred["jam_factor_avg"]
            level      = pred["level"]
            delay_min  = max(0.0, dur_m - free_flow_minutes)
            confidence = pred["confidence"]

        if dur_s < best_dur:
            best_dur = dur_s
            best_idx = idx

        arrival_dt = departure_dt + timedelta(seconds=dur_s)
        slots.append(TimeSlot(
            slot_index        = idx,
            departure_time    = f"{hour:02d}:{minute:02d}",
            arrival_time_est  = f"{arrival_dt.hour:02d}:{arrival_dt.minute:02d}",
            duration_minutes  = round(dur_m, 1),
            jam_factor        = round(jam, 3),
            level             = level,
            delay_minutes     = round(delay_min, 1),
            is_recommended    = False,
            confidence        = round(confidence, 2),
        ))

    # ── 5. Mark recommended slot ──────────────────────────────────────────────
    if slots:
        slots[best_idx] = slots[best_idx].model_copy(update={"is_recommended": True})

    h, m = _time_from_slot(best_idx)

    return PlanDriveResponse(
        slots                  = slots,
        best_slot_index        = best_idx,
        best_departure         = f"{h:02d}:{m:02d}",
        target_date            = req.target_date,
        junctions_found        = len(junction_ids),
        junction_ids           = junction_ids,       # ← جديد
        route_coverage         = coverage,
        source                 = source,
        tomtom_reference_date  = reference_date,
    )