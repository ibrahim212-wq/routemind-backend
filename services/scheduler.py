"""
services/scheduler.py — RouteMind AI Backend

النظام الموحّد:
  - intelligent_scan (scanner.py) هو النظام الوحيد للـ notifications
  - يشتغل فوراً عند بدء الـ instance ثم كل 10 دقائق
    (الـ immediate first-run بيحل مشكلة Cloud Run restarts —
     قبل كده أول scan كان بييجي بعد 15 دقيقة من كل restart،
     فرحلات كتير كانت بتفوت من غير scan خالص)
"""

import os
import logging
import asyncio
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from services.supabase_client import get_supabase

from services.scanner import run_intelligent_scan
from services.pre_trip_collector import run_pre_trip_collection

logger = logging.getLogger("routemind.scheduler")


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

    # كل 10 دقائق + أول تشغيل فوراً (next_run_time=now)
    # كده أي رحلة في الـ window بتتسكان حتى لو الـ instance لسه مبتدي
    scheduler.add_job(
        run_intelligent_scan_sync, "interval",
        minutes=10, id="intelligent_scan",
        next_run_time=datetime.now(),
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        run_pre_trip_collection_sync, "interval",
        minutes=15, id="pre_trip_collection",
        max_instances=1, coalesce=True,
    )

    scheduler.start()
    logger.info("Scheduler started — intelligent_scan every 10 min (first run NOW) ✓")
    return scheduler