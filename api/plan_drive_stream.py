"""
api/plan_drive_stream.py
POST /api/plan-drive-stream

Traffic Level System:
  - Hybrid normalization: jam_base = (ratio - 1.0) / 0.50
  - Levels: FREE < 0.15 | LIGHT < 0.32 | MODERATE < 0.55 | HIGH < 0.80 | VERY_HIGH
  - Duration buffer: 1.0 + (jam_base × 0.50) — بدون DOW boost
  - الخميس: duration multiplier ×1.10 (أوقات) + rush-hour bars boost فقط
  - الجمعة: bars boost +0.08 (مش بيأثر على الـ duration)

jam_base    → buffer + duration فقط
jam_display → bars + level للـ user (بيشمل DOW boost)
"""

import json
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timedelta, date

from model.tier1 import predict_route_duration, jam_to_level
from services.junction_mapper import (
    map_route_to_junctions,
    calculate_route_coverage,
)
from services.google_traffic import stream_google_plan, get_google_free_flow

import logging
logger = logging.getLogger("routemind.plan_drive_stream")

router = APIRouter()

CONF_GOOGLE       = 0.90
CONF_INTERPOLATED = 0.75
CONF_FALLBACK     = 0.55

# DOW boost للجمعة (يُضاف على الـ display jam للـ bars فقط)
DOW_BOOST = {
    4: 0.08,  # الجمعة
}

# Rush hours للخميس (extra boost)
THURSDAY_MORNING_RUSH = (7, 10)
THURSDAY_EVENING_RUSH = (16, 20)
THURSDAY_RUSH_BOOST   = 0.10

# Thursday raw duration multiplier — بيعوّض إن Google بتيجيب أرقام أقل للخميس
DOW_DUR_MULTIPLIER = {
    3: 1.10,  # الخميس: +10% على الـ raw duration قبل حساب الـ jam
}


# ─── Jam helpers ──────────────────────────────────────────────────────────────

def _jam_adaptive(duration_s: int, free_flow_s: int) -> float:
    """
    Hybrid normalization: ratio to free flow.
    +50% زيادة عن الـ free flow = jam 1.0.
    بيخلي HIGH/VERY HIGH يظهروا بس لو فيه زحمة حقيقية.
    """
    ratio = duration_s / max(free_flow_s, 1)
    return round(max(0.0, min(1.0, (ratio - 1.0) / 0.50)), 3)


def _apply_dow_boost(jam: float, target_date: date, dep_hour: int) -> float:
    """
    DOW boost للخميس والجمعة.
    الخميس rush hours بيأخد boost إضافي.
    """
    boost = DOW_BOOST.get(target_date.weekday(), 0.0)

    if target_date.weekday() == 3:  # الخميس
        m_start, m_end = THURSDAY_MORNING_RUSH
        e_start, e_end = THURSDAY_EVENING_RUSH
        if m_start <= dep_hour <= m_end or e_start <= dep_hour <= e_end:
            boost += THURSDAY_RUSH_BOOST

    return round(min(1.0, jam + boost), 3)


def _jam_to_level_google(jam: float) -> str:
    """
    Traffic levels للـ Google results.
    مستقلة عن jam_to_level في tier1.py (اللي بيستخدمه الـ historical shimmer).
    """
    if jam < 0.15: return "FREE"
    if jam < 0.32: return "LIGHT"
    if jam < 0.55: return "MODERATE"
    if jam < 0.80: return "HIGH"
    return "VERY_HIGH"


def _continuous_buffer(jam: float) -> float:
    """
    Continuous buffer يتدرج مع شدة الزحمة:
      jam = 0.0  → buffer = 1.00 (0%)
      jam = 0.32 → buffer = 1.16 (16%)
      jam = 0.55 → buffer = 1.28 (28%)
      jam = 0.80 → buffer = 1.40 (40%)
      jam = 1.00 → buffer = 1.50 (50%)
    """
    return round(1.0 + (jam * 0.50), 3)


# ─── Slot builder ─────────────────────────────────────────────────────────────

def _time_from_slot(slot_idx: int) -> tuple[int, int]:
    total = 5 * 60 + slot_idx * 15
    return total // 60, total % 60


def _slot_idx_from_hm(h: int, m: int) -> int:
    return ((h * 60 + m) - 5 * 60) // 15


def _build_slot_update(
    idx: int,
    raw_dur_s: int,
    target_date: date,
    free_flow_s: int,
    free_flow_min: float,
    confidence: float,
) -> dict:
    hour, minute = _time_from_slot(idx)
    dep = datetime(target_date.year, target_date.month, target_date.day, hour, minute)

    # ① Thursday duration multiplier — يأثر على الـ duration بس
    raw_dur_s = int(raw_dur_s * DOW_DUR_MULTIPLIER.get(target_date.weekday(), 1.0))

    # ② Base jam: ratio to free flow — يُستخدم للـ duration
    jam_base = _jam_adaptive(raw_dur_s, free_flow_s)

    # ③ Display jam: DOW boost للـ bars فقط — مش بيأثر على الـ duration
    jam_display = _apply_dow_boost(jam_base, target_date, hour)

    # ④ Level من jam_display، buffer من jam_base (فصل تام بين الـ bars والـ duration)
    level         = _jam_to_level_google(jam_display)
    buffer        = _continuous_buffer(jam_base)
    buffered_s    = int(raw_dur_s * buffer)
    dur_m         = buffered_s / 60
    delay         = max(0.0, dur_m - free_flow_min)
    arr           = dep + timedelta(seconds=buffered_s)

    return {
        "type":             "update",
        "slot_index":       idx,
        "departure_time":   f"{hour:02d}:{minute:02d}",
        "arrival_time_est": f"{arr.hour:02d}:{arr.minute:02d}",
        "duration_minutes": round(dur_m, 1),
        "jam_factor":       round(jam_display, 3),
        "level":            level,
        "delay_minutes":    round(delay, 1),
        "is_recommended":   False,
        "confidence":       round(confidence, 2),
        "is_loading":       False,
    }


# ─── Request model ────────────────────────────────────────────────────────────

class Waypoint(BaseModel):
    lat: float
    lng: float


class PlanDriveStreamRequest(BaseModel):
    waypoints:             list[Waypoint]
    target_date:           str
    base_duration_seconds: int
    origin_lat:            float
    origin_lng:            float
    dest_lat:              float
    dest_lng:              float


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/plan-drive-stream")
async def plan_drive_stream(req: PlanDriveStreamRequest):

    async def generate():
        target_date = date.fromisoformat(req.target_date)

        waypoints_dicts = [{"lat": w.lat, "lng": w.lng} for w in req.waypoints]
        if not waypoints_dicts:
            waypoints_dicts = [
                {"lat": req.origin_lat, "lng": req.origin_lng},
                {"lat": req.dest_lat,   "lng": req.dest_lng},
            ]

        junctions    = map_route_to_junctions(waypoints_dicts)
        junction_ids = [j["junction_id"] for j in junctions]
        coverage     = calculate_route_coverage(waypoints_dicts)

        logger.info(
            f"Plan drive: base_dur={req.base_duration_seconds}s | "
            f"date={req.target_date} ({target_date.strftime('%A')}) | "
            f"waypoints={len(req.waypoints)}"
        )

        free_flow_min = req.base_duration_seconds / 60

        # ── Step 1: Historical shimmer ──────────────────────────────────────
        best_hist_dur = float("inf")
        best_idx      = 0
        historical_s: dict[int, int] = {}

        for idx in range(76):
            hour, minute = _time_from_slot(idx)
            departure_dt = datetime(
                target_date.year, target_date.month, target_date.day, hour, minute
            )
            pred  = predict_route_duration(
                junction_ids, departure_dt, req.base_duration_seconds
            )
            dur_s = pred["predicted_seconds"]
            historical_s[idx] = dur_s

            if dur_s < best_hist_dur:
                best_hist_dur = dur_s
                best_idx      = idx

            arrival_dt = departure_dt + timedelta(seconds=dur_s)
            yield json.dumps({
                "type":             "slot",
                "slot_index":       idx,
                "departure_time":   f"{hour:02d}:{minute:02d}",
                "arrival_time_est": f"{arrival_dt.hour:02d}:{arrival_dt.minute:02d}",
                "duration_minutes": round(dur_s / 60, 1),
                "jam_factor":       round(pred["jam_factor_avg"], 3),
                "level":            pred["level"],
                "delay_minutes":    round(max(0.0, dur_s/60 - free_flow_min), 1),
                "is_recommended":   False,
                "confidence":       round(pred["confidence"], 2),
                "is_loading":       True,
            }) + "\n"
            await asyncio.sleep(0)

        yield json.dumps({"type": "recommend", "slot_index": best_idx}) + "\n"
        await asyncio.sleep(0)

        # ── Step 2: Google 20 concurrent requests ───────────────────────────
        google_results: dict[int, int] = {}

        async for h, m, dur_s in stream_google_plan(
            req.origin_lat, req.origin_lng,
            req.dest_lat,   req.dest_lng,
            target_date,
        ):
            slot_idx = _slot_idx_from_hm(h, m)
            if 0 <= slot_idx < 76:
                google_results[slot_idx] = dur_s

        # ── Step 3: Hybrid levels + continuous buffer ────────────────────────
        if google_results:
            min_dur = min(google_results.values())   # free_flow_s
            max_dur = max(google_results.values())
            free_flow_min_g = min_dur / 60

            logger.info(
                f"Google: {len(google_results)}/20 | "
                f"min={min_dur}s ({min_dur/60:.1f}m) | "
                f"max={max_dur}s ({max_dur/60:.1f}m) | "
                f"variance={(max_dur-min_dur)/60:.1f}m"
            )

            # Interpolation للـ 76 slots
            anchors = sorted(google_results.items())
            first_i = anchors[0][0]
            last_i  = anchors[-1][0]
            full_slots: dict[int, tuple[int, float]] = {}

            for i in range(first_i, last_i + 1):
                if i in google_results:
                    full_slots[i] = (google_results[i], CONF_GOOGLE)
                else:
                    before = next(((x, d) for x, d in reversed(anchors) if x < i), None)
                    after  = next(((x, d) for x, d in anchors if x > i), None)
                    if before and after:
                        r   = (i - before[0]) / (after[0] - before[0])
                        d_s = int(before[1] + r * (after[1] - before[1]))
                        full_slots[i] = (d_s, CONF_INTERPOLATED)

            for idx in range(76):
                if idx not in full_slots:
                    if idx < first_i:
                        full_slots[idx] = (google_results[first_i], CONF_FALLBACK)
                    else:
                        full_slots[idx] = (google_results[last_i], CONF_FALLBACK)

            # بعت الـ 76 updates
            best_g_dur = float("inf")
            best_g_idx = best_idx

            for idx in range(76):
                raw_dur_s, conf = full_slots[idx]
                update = _build_slot_update(
                    idx, raw_dur_s, target_date,
                    min_dur,
                    free_flow_min_g, conf
                )
                if update["duration_minutes"] * 60 < best_g_dur:
                    best_g_dur = update["duration_minutes"] * 60
                    best_g_idx = idx

                yield json.dumps(update) + "\n"
                await asyncio.sleep(0)

            yield json.dumps({"type": "recommend", "slot_index": best_g_idx}) + "\n"

        else:
            # Fallback: historical
            logger.warning("Google failed → historical fallback")
            for idx in range(76):
                hour, minute = _time_from_slot(idx)
                dep = datetime(target_date.year, target_date.month,
                               target_date.day, hour, minute)
                dur_s = historical_s[idx]
                arr   = dep + timedelta(seconds=dur_s)
                yield json.dumps({
                    "type":             "update",
                    "slot_index":       idx,
                    "departure_time":   f"{hour:02d}:{minute:02d}",
                    "arrival_time_est": f"{arr.hour:02d}:{arr.minute:02d}",
                    "duration_minutes": round(dur_s / 60, 1),
                    "jam_factor":       0.0,
                    "level":            "FREE",
                    "delay_minutes":    0.0,
                    "is_recommended":   False,
                    "confidence":       CONF_FALLBACK,
                    "is_loading":       False,
                }) + "\n"
                await asyncio.sleep(0)

        # ── Done ────────────────────────────────────────────────────────────
        yield json.dumps({
            "type":           "done",
            "junction_ids":   junction_ids,
            "route_coverage": coverage,
            "source":         "google" if google_results else "historical",
            "google_slots":   len(google_results),
        }) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
        },
    )