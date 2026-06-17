"""
api/scan_route.py
POST /api/scan-route

Called by Firebase Cloud Functions every 15 minutes for active planned trips.
Implements the "smart" monitoring logic:
  - Builds route timeline (when user arrives at each junction)
  - Checks if congestion will be there WHEN the user arrives
  - Only alerts if delay > 5 min (ignores transient congestion)
  - Returns recommendation: leave early / all clear / slight delay
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import logging

from model.tier1 import get_historical_jam, jam_to_level, jam_to_multiplier
from model.tier2 import predict_junction_short_horizon
from services.tomtom import get_readings_for_junctions
from services.junction_mapper import get_junction_by_id
from services.supabase_client import get_supabase
from services.google_traffic import get_google_route_congestion

logger = logging.getLogger("routemind.scan_route")
router = APIRouter()


# ── Request / Response ────────────────────────────────────────────────────────
class ScanRouteRequest(BaseModel):
    trip_id:              str
    user_id:              str
    junction_ids:         list[str]       # Route junctions in order
    departure_time:       str             # ISO: "2026-05-20T08:50:00"
    scan_time:            str             # ISO: "2026-05-20T07:55:00"
    base_duration_seconds: int
    # History from Firestore (last 12 readings per junction)
    junction_history:     Optional[dict]  # {junction_id: [{reading}, ...]}


class JunctionAlert(BaseModel):
    junction_id:      str
    arrives_at:       str          # When user will be at this junction
    predicted_jam:    float
    level:            str
    delay_added_min:  float


class ScanRouteResponse(BaseModel):
    trip_id:           str
    should_alert:      bool
    alert_type:        str          # "leave_early" | "slight_delay" | "all_clear"
    total_delay_min:   float
    leave_early_by_min: float
    message:           str
    message_ar:        str          # Arabic message for the notification
    congested_junctions: list[JunctionAlert]
    scan_time:         str


# ── Smart Timeline Builder ────────────────────────────────────────────────────
def _build_route_timeline(
    junction_ids: list[str],
    departure_dt: datetime,
    base_duration_seconds: int,
) -> list[dict]:
    """
    For each junction, calculates WHEN the user will arrive there.
    Uses historical jam to estimate progressive delays.
    """
    if not junction_ids:
        return []

    per_junction_s = base_duration_seconds / len(junction_ids)
    timeline = []
    elapsed_s = 0.0

    for jid in junction_ids:
        midpoint_s    = elapsed_s + per_junction_s * 0.5
        arrival_dt    = departure_dt + timedelta(seconds=midpoint_s)
        hist_jam      = get_historical_jam(jid, arrival_dt)
        multiplier    = jam_to_multiplier(hist_jam)
        seg_duration  = per_junction_s * multiplier
        elapsed_s    += seg_duration

        timeline.append({
            "junction_id": jid,
            "arrival_dt":  arrival_dt,
            "hist_jam":    hist_jam,
        })

    return timeline


# ── The Key Question: Is Congestion There WHEN User Arrives? ─────────────────
def _will_congestion_affect_user(
    junction_id: str,
    user_arrives_at: datetime,
    current_jam: float,
    junction_history: dict,
) -> dict:
    """
    Smart check: will the current congestion still be there when the user arrives?

    Uses:
    1. LSTM prediction if history available (Tier 2)
    2. Historical average comparison (Tier 1 fallback)
    3. The KEY logic: "is this anomalous AND will it persist?"
    """
    hist_jam = get_historical_jam(junction_id, user_arrives_at)

    # Try Tier 2 (LSTM) if history available
    if junction_history and junction_id in junction_history:
        hist_readings = junction_history[junction_id]
        if len(hist_readings) >= 6:  # Need at least 6 readings
            readings_snaps = [r.get("all_junctions", {}) for r in hist_readings]
            timestamps     = [
                datetime.fromisoformat(r["timestamp"])
                for r in hist_readings
            ]
            tier2_result = predict_junction_short_horizon(
                junction_id, readings_snaps, timestamps
            )
            if tier2_result:
                # Use LSTM prediction at the appropriate horizon
                minutes_until = (user_arrives_at - datetime.utcnow()).total_seconds() / 60
                if minutes_until <= 20:
                    predicted_jam = tier2_result["jam_15min"]
                elif minutes_until <= 45:
                    predicted_jam = tier2_result["jam_30min"]
                else:
                    predicted_jam = tier2_result["jam_60min"]

                return {
                    "predicted_jam": predicted_jam,
                    "source": "lstm",
                    "is_anomalous": current_jam > hist_jam + 0.15,
                }

    # Tier 1 fallback: check if current reading is anomalous vs historical
    is_anomalous  = current_jam > hist_jam + 0.15
    # Simple persistence model: congestion decays toward historical over time
    minutes_until = max(0, (user_arrives_at - datetime.utcnow()).total_seconds() / 60)
    decay_factor  = max(0.0, 1.0 - minutes_until / 60.0)  # fully decays in 60 min
    predicted_jam = hist_jam + (current_jam - hist_jam) * decay_factor

    return {
        "predicted_jam": predicted_jam,
        "source": "tier1_decay",
        "is_anomalous": is_anomalous,
    }


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/scan-route", response_model=ScanRouteResponse)
async def scan_route(req: ScanRouteRequest):
    """
    Smart route scan. Called by Firebase Cloud Functions every 15 minutes.
    """
    departure_dt = datetime.fromisoformat(req.departure_time)
    scan_dt      = datetime.fromisoformat(req.scan_time)

    # 1. Get current TomTom readings for all route junctions
    junction_meta = []
    for jid in req.junction_ids:
        j = get_junction_by_id(jid)
        if j:
            junction_meta.append(j)

    logger.info(f"Scanning trip {req.trip_id}: {len(junction_meta)} junctions")
    current_readings = await get_readings_for_junctions(junction_meta)

    # 2. Build route timeline: when does user arrive at each junction?
    timeline = _build_route_timeline(
        req.junction_ids, departure_dt, req.base_duration_seconds
    )

    # 3. For each junction: "will congestion be there WHEN user arrives?"
    free_flow_min = req.base_duration_seconds / 60
    total_predicted_s = 0.0
    alerted_junctions = []

    for entry in timeline:
        jid          = entry["junction_id"]
        arrives_at   = entry["arrival_dt"]
        current_jam  = (
            current_readings[jid]["jam_factor"] / 10.0
            if jid in current_readings else entry["hist_jam"]
        )

        # THE KEY CHECK: will this congestion be there when user arrives?
        check = _will_congestion_affect_user(
            jid, arrives_at, current_jam, req.junction_history
        )
        predicted_jam = check["predicted_jam"]
        delay_min     = max(0.0, (jam_to_multiplier(predicted_jam) - 1.0) *
                           (req.base_duration_seconds / len(req.junction_ids) / 60))

        total_predicted_s += (req.base_duration_seconds / len(req.junction_ids)) * jam_to_multiplier(predicted_jam)

        if predicted_jam >= 0.3 and delay_min >= 1.0:
            alerted_junctions.append(JunctionAlert(
                junction_id     = jid,
                arrives_at      = arrives_at.strftime("%H:%M"),
                predicted_jam   = round(predicted_jam, 3),
                level           = jam_to_level(predicted_jam),
                delay_added_min = round(delay_min, 1),
            ))

    # 4. Decision: should we alert the user?
    # Headline delay = the REAL Google delay (live − free-flow) — the same signal
    # the background scanner + notifications use — NOT the inflated historical
    # model sum. Fall back to the model estimate only if the Google/trip lookup
    # is unavailable, so the endpoint never breaks.
    model_delay_min = max(0.0, total_predicted_s / 60 - free_flow_min)
    total_delay_min = model_delay_min
    try:
        supabase = get_supabase()
        trip_res = (supabase.table("planned_trips").select("*")
                    .eq("id", req.trip_id).single().execute())
        trip = trip_res.data if trip_res else None
        if trip:
            cong = await get_google_route_congestion(
                float(trip["origin_lat"]), float(trip["origin_lng"]),
                float(trip["dest_lat"]),   float(trip["dest_lng"]),
                free_flow_seconds=req.base_duration_seconds,
            )
            if cong:
                total_delay_min = max(
                    0.0, round((cong["live_seconds"] - cong["free_flow_seconds"]) / 60, 1)
                )
    except Exception as e:
        logger.warning(f"scan-route real-delay lookup failed, using model estimate: {e}")
    leave_early_by  = round(total_delay_min * 1.1, 0)  # slight buffer

    if total_delay_min >= 15:
        alert_type = "leave_early"
        should_alert = True
        message = f"Traffic ahead may add {round(total_delay_min)} min to your trip. Leave {round(leave_early_by)} min early."
        message_ar = f"زحمة في طريقك ستضيف {round(total_delay_min)} دقيقة. اتحرك مبكراً بـ {round(leave_early_by)} دقيقة."
    elif total_delay_min >= 5:
        alert_type = "slight_delay"
        should_alert = True
        message = f"Slight traffic detected. Expect {round(total_delay_min)} min delay."
        message_ar = f"زحمة خفيفة في طريقك. توقع تأخير {round(total_delay_min)} دقيقة."
    else:
        alert_type = "all_clear"
        should_alert = False
        message = "Route looks clear. You're good to go!"
        message_ar = "الطريق واضح. اتحرك في معادك!"

    return ScanRouteResponse(
        trip_id              = req.trip_id,
        should_alert         = should_alert,
        alert_type           = alert_type,
        total_delay_min      = round(total_delay_min, 1),
        leave_early_by_min   = leave_early_by,
        message              = message,
        message_ar           = message_ar,
        congested_junctions  = alerted_junctions,
        scan_time            = scan_dt.isoformat(),
    )