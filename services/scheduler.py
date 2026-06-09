"""
services/scheduler.py — Smart Notification Engine
RouteMind AI Backend
"""

import os
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from apscheduler.schedulers.background import BackgroundScheduler
from services.supabase_client import get_supabase, SupabaseClient

from services.scanner import run_intelligent_scan
from services.pre_trip_collector import run_pre_trip_collection

logger = logging.getLogger("routemind.scheduler")


# ── Alert Levels ──────────────────────────────────────────────────────────────

class AlertLevel(Enum):
    CRITICAL  = "critical"
    SERIOUS   = "serious"
    WARNING   = "warning"
    ADVISORY  = "advisory"
    ALL_CLEAR = "all_clear"


# ── Message Library ───────────────────────────────────────────────────────────

MESSAGES = {
    AlertLevel.CRITICAL: [
        {
            "title": "Leave {delay} min early — traffic ahead",
            "body":  "RouteMind AI predicts heavy congestion on your {time} route to {dest}. Tap to see affected roads.",
        },
        {
            "title": "Significant delay on your route",
            "body":  "AI analysis shows {delay} min of congestion to {dest}. Leaving by {new_time} keeps you on schedule.",
        },
        {
            "title": "Your route needs attention",
            "body":  "Heavy traffic detected for your {time} trip to {dest}. Consider leaving {delay} min earlier.",
        },
    ],
    AlertLevel.SERIOUS: [
        {
            "title": "Leave a bit earlier — {delay} min delay",
            "body":  "RouteMind sees congestion building on your route to {dest}. Tap for live conditions.",
        },
        {
            "title": "Traffic on your route to {dest}",
            "body":  "AI predicts a {delay} min delay for your {time} departure. Leaving early helps.",
        },
        {
            "title": "Route update — {delay} min slower",
            "body":  "Congestion detected ahead of your trip to {dest}. Tap to view route details.",
        },
    ],
    AlertLevel.WARNING: [
        {
            "title": "Heads up — slight delay ahead",
            "body":  "Expect around {delay} min of traffic to {dest}. Your {time} trip may be affected.",
        },
        {
            "title": "Traffic building on your route",
            "body":  "RouteMind detected a {delay} min slowdown to {dest}. Tap for details.",
        },
        {
            "title": "Minor congestion to {dest}",
            "body":  "AI sees {delay} min of traffic on your {time} route. You may want to leave slightly early.",
        },
    ],
    AlertLevel.ADVISORY: [
        {
            "title": "Light traffic on your route",
            "body":  "A small {delay} min delay is expected to {dest}. Roads are mostly clear.",
        },
        {
            "title": "Minimal congestion detected",
            "body":  "RouteMind sees {delay} min of light traffic ahead. Your {time} trip looks good.",
        },
    ],
    AlertLevel.ALL_CLEAR: [
        {
            "title": "All clear — you are good to go",
            "body":  "No significant traffic detected. Your {time} trip to {dest} looks smooth.",
        },
        {
            "title": "Roads are clear ahead",
            "body":  "RouteMind confirms light traffic for your {time} departure to {dest}. Leave as planned.",
        },
        {
            "title": "Route confirmed — no delays",
            "body":  "AI analysis shows clear roads to {dest}. Your scheduled departure time is optimal.",
        },
    ],
}


# ── Smart Threshold Engine ────────────────────────────────────────────────────

def get_alert_level(delay_min: float, minutes_to_departure: float) -> AlertLevel | None:
    urgency = max(0.5, 1.0 - minutes_to_departure / 90.0)

    if   delay_min >= 20 * urgency: return AlertLevel.CRITICAL
    elif delay_min >= 12 * urgency: return AlertLevel.SERIOUS
    elif delay_min >=  6 * urgency: return AlertLevel.WARNING
    elif delay_min >=  3 * urgency: return AlertLevel.ADVISORY
    elif minutes_to_departure <= 12: return AlertLevel.ALL_CLEAR
    return None


def compose_notification(
    level: AlertLevel,
    dest: str,
    delay: int,
    leave_by: datetime,
    trip_id: str,
) -> dict:
    templates  = MESSAGES[level]
    seed       = hash(trip_id + str(datetime.now().date())) % len(templates)
    t          = templates[seed]
    fmt_time   = leave_by.strftime("%I:%M %p")
    new_leave  = leave_by - timedelta(minutes=delay)
    fmt_new    = new_leave.strftime("%I:%M %p")

    title = t["title"].format(delay=delay, dest=dest, time=fmt_time, new_time=fmt_new)
    body  = t["body"].format( delay=delay, dest=dest, time=fmt_time, new_time=fmt_new)

    return {
        "title": title,
        "body":  body,
        "data": {
            "trip_id":       trip_id,
            "alert_level":   level.value,
            "delay_minutes": str(delay),
            "tap_action":    "show_route_details",
        },
    }


# ── FCM Sender ────────────────────────────────────────────────────────────────

def send_fcm(user_id: str, notification: dict):
    try:
        from firebase_admin import messaging

        supabase = get_supabase()
        if not supabase:
            return

        res   = supabase.table("fcm_tokens").select("token").eq("user_id", user_id).limit(1).execute()
        token = res.data[0]["token"] if res.data else None
        if not token:
            logger.warning(f"No FCM token for {user_id}")
            return

        messaging.send(messaging.Message(
            notification=messaging.Notification(
                title=notification["title"],
                body=notification["body"],
            ),
            data=notification["data"],
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="routemind_trips",
                    click_action="FLUTTER_NOTIFICATION_CLICK",
                ),
            ),
            token=token,
        ))
        logger.info(f"FCM → {user_id}: {notification['title']}")

    except Exception as e:
        logger.error(f"FCM error: {e}")


# ── Trip Processor ────────────────────────────────────────────────────────────

def _process_trip(trip: dict, now: datetime, supabase):
    try:
        from services.tomtom import get_readings_for_junctions
        from model.tier1 import get_historical_jam, jam_to_multiplier
        from services.junction_mapper import get_junction_by_id

        trip_id       = trip["id"]
        user_id       = trip["user_id"]
        leave_by      = datetime.fromisoformat(trip["leave_by"])
        dest_name     = trip["dest_name"]
        junction_ids  = trip.get("junction_ids") or []
        base_secs     = trip.get("base_duration_seconds", 1800)
        alert_count   = trip.get("alert_count", 0)
        final_sent    = trip.get("final_clear_sent", False)
        alert_sent_at = trip.get("alert_sent_at")

        mins_to_depart = (leave_by - now).total_seconds() / 60

        if alert_count >= 1 and mins_to_depart > 12:
            return

        if alert_sent_at:
            last_sent = datetime.fromisoformat(alert_sent_at)
            if (now - last_sent).total_seconds() < 25 * 60:
                return

        junctions = [j for j in (get_junction_by_id(jid) for jid in junction_ids) if j]
        loop      = asyncio.new_event_loop()
        readings  = loop.run_until_complete(get_readings_for_junctions(junctions))
        loop.close()

        free_min = base_secs / 60
        per_j    = base_secs / max(len(junction_ids), 1)
        total_s  = 0.0

        for i, jid in enumerate(junction_ids):
            arrival_dt  = now + timedelta(seconds=i * per_j)
            hist_jam    = get_historical_jam(jid, arrival_dt)
            current_jam = readings[jid]["jam_factor"] / 10.0 if jid in readings else hist_jam
            decay       = max(0.0, 1.0 - mins_to_depart / 60.0)
            pred_jam    = hist_jam + (current_jam - hist_jam) * decay
            total_s    += per_j * jam_to_multiplier(pred_jam)

        delay_min = max(0.0, total_s / 60 - free_min)

        level = get_alert_level(delay_min, mins_to_depart)
        if level is None:
            return

        if level == AlertLevel.ALL_CLEAR and final_sent:
            return

        if level != AlertLevel.ALL_CLEAR and alert_count >= 1:
            return

        notification = compose_notification(
            level, dest_name, round(delay_min), leave_by, trip_id
        )
        send_fcm(user_id, notification)

        update = {
            "alert_sent_at": now.isoformat(),
            "alert_count":   alert_count + 1,
        }
        if level == AlertLevel.ALL_CLEAR:
            update["final_clear_sent"] = True

        supabase.table("planned_trips").update(update).eq("id", trip_id).execute()

    except Exception as e:
        logger.error(f"Trip {trip.get('id')} error: {e}")


# ── Main Scan Job ─────────────────────────────────────────────────────────────

def scan_trips():
    supabase = get_supabase()
    if not supabase:
        return

    now        = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=1)

    try:
        trips = supabase.table("planned_trips")\
            .select("*")\
            .eq("status", "active")\
            .gte("leave_by", now.isoformat())\
            .lte("leave_by", window_end.isoformat())\
            .execute().data or []

        logger.info(f"Scanning {len(trips)} active trips")
        for trip in trips:
            _process_trip(trip, now, supabase)

    except Exception as e:
        logger.error(f"Scan error: {e}")


# ── Async Wrappers ────────────────────────────────────────────────────────────

def run_intelligent_scan_sync():
    supabase = get_supabase()
    if not supabase:
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_intelligent_scan(supabase))
        loop.close()
    except Exception as e:
        logger.error(f"Intelligent scan error: {e}")


def run_pre_trip_collection_sync():
    supabase = get_supabase()
    if not supabase:
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_pre_trip_collection(supabase))
        loop.close()
    except Exception as e:
        logger.error(f"Pre-trip collection error: {e}")


# ── Scheduler Startup ─────────────────────────────────────────────────────────

def start_scheduler():
    if not get_supabase():
        logger.warning("Supabase not configured — scheduler disabled")
        return None

    scheduler = BackgroundScheduler(timezone="Africa/Cairo")

    scheduler.add_job(scan_trips,                    "interval", minutes=15, id="scan_trips")
    scheduler.add_job(run_intelligent_scan_sync,     "interval", minutes=15, id="intelligent_scan")
    scheduler.add_job(run_pre_trip_collection_sync,  "interval", minutes=15, id="pre_trip_collection")

    scheduler.start()
    logger.info("Scheduler started — 3 jobs every 15 min ✓")
    return scheduler