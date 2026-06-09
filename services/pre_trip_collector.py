"""
services/pre_trip_collector.py

Smart pre-trip data collection.

يجمع readings للـ junctions على طريق اليوزر قبل الرحلة بساعتين،
بطريقة ذكية توفر TomTom calls بدون ما تأثر على الـ prediction.

الـ Logic:
  - لو عندنا data كافية من أسابيع سابقة → نتأكد بسكان بسيط
  - لو عندنا بعض الـ data → نكمل الناقص بس
  - لو مفيش حاجة → نجمع كامل
  - كل reading بتتخزن وبتفيد كل الـ users اللي بيعدوا من نفس الطريق
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import httpx
from services.supabase_client import SupabaseClient

logger = logging.getLogger("routemind.pre_trip")

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4"
    "/flowSegmentData/absolute/10/json"
)

# ─── Thresholds ───────────────────────────────────────────────────────────────
TARGET_READINGS       = 8     # عايزين 8 readings قبل الرحلة
HISTORICAL_MIN        = 10    # لو عندنا ≥ 10 من التاريخ → pattern check
PATTERN_TOLERANCE     = 0.20  # ± 20% في الـ jam → pattern مشابه
COLLECTION_WINDOW_MIN = 30    # بدأ نجمع لو الرحلة بعد 30 دقيقة لـ 3 ساعات
COLLECTION_WINDOW_MAX = 180


# ─── TomTom fetch ────────────────────────────────────────────────────────────

async def _fetch_flow(lat: float, lon: float) -> Optional[Dict]:
    api_key = os.getenv("TOMTOM_API_KEY", "")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                TOMTOM_FLOW_URL,
                params={"point": f"{lat},{lon}", "key": api_key},
            )
        if resp.status_code != 200:
            return None
        data = resp.json().get("flowSegmentData", {})
        if not data or data.get("roadClosure"):
            return None

        current   = float(data.get("currentSpeed", 0))
        free_flow = float(data.get("freeFlowSpeed", 1))
        if free_flow <= 0:
            return None

        cr = max(0.0, 1.0 - current / free_flow)
        return {
            "jam_factor":       round(cr * 10, 2),
            "current_speed":    round(current, 1),
            "free_flow_speed":  round(free_flow, 1),
            "congestion_ratio": round(cr, 3),
            "speed_reduction":  round(free_flow - current, 1),
            "confidence":       round(float(data.get("confidence", 0.5)), 2),
        }
    except Exception:
        return None


# ─── Supabase queries ────────────────────────────────────────────────────────

def _get_recent_readings(
    junction_id: str,
    hours: int,
    supabase: SupabaseClient,
) -> List[Dict]:
    """آخر readings من الـ x ساعات الماضية."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        res = (supabase.table("junction_readings")
               .select("jam_factor, recorded_at")
               .eq("junction_id", junction_id)
               .gte("recorded_at", cutoff)
               .order("recorded_at", desc=True)
               .execute())
        return res.data or []
    except Exception:
        return []


def _get_historical_same_slot(
    junction_id: str,
    leave_time: datetime,
    weeks_back: int = 2,
    supabase: SupabaseClient = None,
) -> List[Dict]:
    """
    readings من نفس اليوم الأسبوع ونفس الساعة من آخر أسبوعين.
    يغطي ±30 دقيقة حول الوقت المطلوب.
    """
    results = []
    dow  = leave_time.weekday()
    hour = leave_time.hour

    for w in range(1, weeks_back + 1):
        # نفس اليوم الأسبوع من الأسبوع w الفايت
        past = leave_time - timedelta(weeks=w)
        past_start = past.replace(minute=0,  second=0, microsecond=0) - timedelta(minutes=30)
        past_end   = past.replace(minute=59, second=59) + timedelta(minutes=30)

        try:
            res = (supabase.table("junction_readings")
                   .select("jam_factor, recorded_at")
                   .eq("junction_id", junction_id)
                   .gte("recorded_at", past_start.isoformat())
                   .lte("recorded_at", past_end.isoformat())
                   .execute())
            results.extend(res.data or [])
        except Exception:
            continue

    return results


def _is_pattern_similar(
    current_jam: float,
    historical: List[Dict],
    tolerance: float = PATTERN_TOLERANCE,
) -> bool:
    """هل الـ jam الحالي قريب من المتوسط التاريخي؟"""
    if not historical:
        return False
    hist_avg = sum(float(r["jam_factor"]) for r in historical) / len(historical)
    if hist_avg == 0:
        return current_jam < 1.0
    diff = abs(current_jam - hist_avg) / hist_avg
    return diff <= tolerance


async def _store_reading(
    junction_id: str,
    flow: Dict,
    supabase: SupabaseClient,
) -> None:
    now = datetime.now(timezone.utc)
    try:
        supabase.table("junction_readings").insert({
            "junction_id":      junction_id,
            "recorded_at":      now.isoformat(),
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
            "scan_trigger":     "pre_trip_collection",
        }).execute()
    except Exception as e:
        logger.error(f"Failed to store reading for {junction_id}: {e}")


# ─── Smart collection decision ───────────────────────────────────────────────

async def smart_collect_junction(
    junction_id: str,
    lat: float,
    lon: float,
    leave_time: datetime,
    supabase: SupabaseClient,
) -> str:
    """
    يقرر بذكاء كام reading نجيب من TomTom لـ junction معين.

    Returns:
        "skipped"    → عندنا data كافية ومش محتاجين
        "validated"  → عملنا scan للتأكد بس
        "collected"  → جمعنا reading جديدة
        "no_data"    → TomTom مش رد
    """
    # ── 1. الـ recent readings من اليوم ده ────────────────────────
    recent = _get_recent_readings(junction_id, hours=2, supabase=supabase)

    if len(recent) >= TARGET_READINGS:
        # عندنا readings كافية من اليوم ده → skip
        logger.debug(f"{junction_id}: {len(recent)} recent readings → skip")
        return "skipped"

    # ── 2. الـ historical من أسابيع سابقة ─────────────────────────
    historical = _get_historical_same_slot(junction_id, leave_time, supabase=supabase)

    # ── 3. قرار الجمع ──────────────────────────────────────────────
    if len(historical) >= HISTORICAL_MIN and len(recent) >= 2:
        # عندنا data تاريخية + شوية recent
        # → نعمل scan واحد للتأكد إن الـ pattern نفسه

        flow = await _fetch_flow(lat, lon)
        if flow is None:
            return "no_data"

        if _is_pattern_similar(flow["jam_factor"], historical):
            # Pattern مشابه → نخزن الـ reading ده وبس
            await _store_reading(junction_id, flow, supabase)
            logger.debug(
                f"{junction_id}: pattern similar "
                f"(current={flow['jam_factor']:.1f}) → 1 scan only"
            )
            return "validated"
        else:
            # Pattern اختلف → نجمع reading ونرجع في الـ scan الجاي
            await _store_reading(junction_id, flow, supabase)
            logger.info(
                f"{junction_id}: pattern changed "
                f"(current={flow['jam_factor']:.1f}) → collecting more"
            )
            return "collected"

    else:
        # مفيش data كافية → نجمع reading جديدة
        # (الـ scheduler بيشتغل كل 15 دقيقة، كل مرة بتجيب reading)
        flow = await _fetch_flow(lat, lon)
        if flow is None:
            return "no_data"

        await _store_reading(junction_id, flow, supabase)
        logger.debug(
            f"{junction_id}: collecting "
            f"({len(recent)}/{TARGET_READINGS} readings)"
        )
        return "collected"


# ─── Main entry point ─────────────────────────────────────────────────────────

async def run_pre_trip_collection(supabase: SupabaseClient) -> None:
    """
    يشتغل من الـ scheduler كل 15 دقيقة.

    يجمع readings للـ trips اللي مبدأها بعد 30 دقيقة → 3 ساعات.
    كل reading بتتخزن وبتفيد كل الـ users اللي بيعدوا من نفس الطريق.
    """
    from model.loader import ModelLoader

    now      = datetime.now(timezone.utc)
    min_time = (now + timedelta(minutes=COLLECTION_WINDOW_MIN)).isoformat()
    max_time = (now + timedelta(minutes=COLLECTION_WINDOW_MAX)).isoformat()

    try:
        res = (supabase.table("planned_trips")
               .select("id, junction_ids, leave_by")
               .eq("status", "active")
               .gte("leave_by", min_time)
               .lte("leave_by", max_time)
               .execute())
    except Exception as e:
        logger.error(f"Pre-trip collection: failed to fetch trips: {e}")
        return

    trips = res.data or []
    if not trips:
        return

    # ── جمع كل الـ unique junctions من كل الـ trips ──────────────────
    # لو أكتر من user بيعدوا من نفس الـ junction → نجمعه مرة واحدة بس
    unique_junctions: Dict[str, datetime] = {}
    for trip in trips:
        leave_time = datetime.fromisoformat(trip["leave_by"])
        for jid in (trip.get("junction_ids") or []):
            if jid not in unique_junctions:
                unique_junctions[jid] = leave_time

    if not unique_junctions:
        return

    logger.info(
        f"Pre-trip collection: {len(trips)} trip(s), "
        f"{len(unique_junctions)} unique junctions"
    )

    combined_meta = ModelLoader.get_combined_meta()
    stats = {"skipped": 0, "validated": 0, "collected": 0, "no_data": 0}

    for junction_id, leave_time in unique_junctions.items():
        row = combined_meta[combined_meta["junction_id"] == junction_id]
        if row.empty:
            continue

        r   = row.iloc[0]
        lat = float(r["latitude"])
        lon = float(r["longitude"])

        result = await smart_collect_junction(
            junction_id=junction_id,
            lat=lat,
            lon=lon,
            leave_time=leave_time,
            supabase=supabase,
        )
        stats[result] = stats.get(result, 0) + 1

    logger.info(
        f"Pre-trip collection done: "
        f"skipped={stats['skipped']} | "
        f"validated={stats['validated']} | "
        f"collected={stats['collected']} | "
        f"no_data={stats['no_data']}"
    )