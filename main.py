"""
RouteMind AI Backend — FastAPI
==============================
Deploy on Google Cloud Run.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from services.scheduler import start_scheduler
import json
import os

from model.loader import ModelLoader
from api.plan_drive import router as plan_drive_router
from api.scan_route import router as scan_route_router
from api.route_compare import router as route_compare_router
from api.plan_drive_stream import router as plan_drive_stream_router
from api.trip_alert import router as trip_alert_router
from api.assistant import router as assistant_router
from api.places import router as places_router
from api.copilot import router as copilot_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routemind")


def init_firebase():
    try:
        import firebase_admin
        from firebase_admin import credentials

        # Service-account Certificate credential from Secret Manager. This is the
        # battle-tested firebase-admin path (key-based JWT bearer exchange), which
        # — unlike metadata-server ADC — reliably obtains the firebase.messaging
        # scope required for FCM v1 delivery.
        creds_json = os.environ.get("FIREBASE_CREDENTIALS")
        if creds_json:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized from Secret Manager ✓")
            return

        # Explicit key file fallback (only when GOOGLE_APPLICATION_CREDENTIALS is
        # set) — no implicit default path so a stray bundled key can't load.
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path and os.path.exists(key_path):
            firebase_admin.initialize_app(credentials.Certificate(key_path))
            logger.info("Firebase initialized from key file ✓")
            return

        logger.warning("Firebase credentials not found — notifications disabled")

    except Exception as e:
        logger.warning(f"Firebase init failed: {e} — notifications disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RouteMind AI model...")
    ModelLoader.load()
    logger.info("Model ready ✓")
    init_firebase()
    start_scheduler()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RouteMind AI Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plan_drive_router,        prefix="/api")
app.include_router(scan_route_router,        prefix="/api")
app.include_router(route_compare_router,     prefix="/api")
app.include_router(plan_drive_stream_router, prefix="/api")
app.include_router(trip_alert_router,        prefix="/api")
app.include_router(assistant_router,         prefix="/api")
app.include_router(places_router,            prefix="/api")
app.include_router(copilot_router,           prefix="/api")


@app.post("/api/test-notify")
async def test_notify(body: dict):
    """
    Test endpoint — يبعت notification فورية لـ user معين.
    للتأكد إن الـ FCM pipeline شغال.
    """
    try:
        from firebase_admin import messaging
        from services.supabase_client import get_supabase

        user_id = body.get("user_id")
        if not user_id:
            return {"status": "error", "reason": "user_id مطلوب"}

        supabase = get_supabase()
        if not supabase:
            return {"status": "error", "reason": "Supabase مش متصل"}

        res   = supabase.table("fcm_tokens").select("token").eq("user_id", user_id).limit(1).execute()
        token = res.data[0]["token"] if res.data else None

        if not token:
            return {"status": "error", "reason": f"مفيش FCM token للـ user {user_id}"}

        message_name = messaging.send(messaging.Message(
            notification=messaging.Notification(
                title="RouteMind Test 🚦",
                body="الـ backend شغال وبيبعت notifications ✅",
            ),
            data={
                "type":       "test",
                "tap_action": "show_route_details",
            },
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="routemind_trips",
                    click_action="FLUTTER_NOTIFICATION_CLICK",
                ),
            ),
            token=token,
        ))

        return {
            "status":        "sent",
            "user_id":       user_id,
            "message_name":  message_name,
            "token_preview": token[:20] + "...",
        }

    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.post("/api/run-scan")
async def run_scan_now():
    """
    بيشغّل الـ intelligent scan فوراً (بدل انتظار الـ scheduler).
    للاختبار + ممكن يتربط بـ Cloud Scheduler كضمان إضافي.
    """
    try:
        from services.scanner import run_intelligent_scan
        from services.supabase_client import get_supabase

        supabase = get_supabase()
        if not supabase:
            return {"status": "error", "reason": "Supabase مش متصل"}

        await run_intelligent_scan(supabase)
        return {"status": "scan_completed"}

    except Exception as e:
        logger.error(f"Manual scan failed: {e}")
        return {"status": "error", "reason": str(e)}


@app.post("/api/scan-now/{trip_id}")
async def scan_now(trip_id: str):
    """
    Immediately scan ONE trip — the app should call this right after creating a
    trip so short-notice trips (leaving soon) don't wait for the next scheduler
    cycle. Runs the same scan_trip path as the scheduler.
    """
    try:
        from services.scanner import scan_trip
        from services.supabase_client import get_supabase

        supabase = get_supabase()
        if not supabase:
            return {"status": "error", "reason": "Supabase مش متصل"}

        res  = supabase.table("planned_trips").select("*").eq("id", trip_id).single().execute()
        trip = res.data
        if not trip:
            return {"status": "error", "reason": f"trip {trip_id} not found"}

        # reverse_adj is unused by the route-level scan; pass an empty dict.
        await scan_trip(trip, {}, supabase)
        return {"status": "scanned", "trip_id": trip_id}

    except Exception as e:
        logger.error(f"scan-now failed for {trip_id}: {e}")
        return {"status": "error", "reason": str(e)}


@app.post("/demo/scan-now")
async def demo_scan_now():
    """
    DEMO TRIGGER — no new pipeline. Picks the most-congested of a few common Cairo
    routes RIGHT NOW, creates a temporary planned trip (departure = now, correct
    Google free-flow baseline), runs the EXISTING scan_trip, then broadcasts the
    EXISTING congestion notification (same format) to ALL registered devices by
    calling the existing _send_congestion_notification once per user that has
    tokens. The notification carries trip_id, so tapping opens the existing alert
    screen with the per-junction rows this scan just wrote.
    """
    import math
    from datetime import datetime, timezone, timedelta
    from services.supabase_client import get_supabase
    from services.google_traffic import get_google_route_congestion
    from services.scanner import (
        scan_trip, _send_trip_alert_notification, _jam_to_level, _haversine_km,
    )
    from model.loader import ModelLoader

    supabase = get_supabase()
    if not supabase:
        return {"status": "error", "reason": "Supabase not connected"}

    # ── a. pick the most-congested route right now (existing traffic source) ──
    routes = [
        ("Ring Road",           30.0558, 31.3400, 30.0850, 31.3050),
        ("Route 75M (New Cairo)", 30.0270, 31.4380, 29.9980, 31.4760),
        ("Salah Salem",         30.0695, 31.2843, 30.0444, 31.2357),
    ]
    best = None
    for name, a, b, c, d in routes:
        cong = await get_google_route_congestion(a, b, c, d, free_flow_seconds=None, departure_utc=None)
        if not cong:
            continue
        if best is None or cong["congestion_ratio"] > best[1]["congestion_ratio"]:
            best = (name, cong, (a, b, c, d))
    if best is None:
        return {"status": "error", "reason": "no live route data"}
    name, cong, (a, b, c, d) = best
    base_dur = int(cong["free_flow_seconds"])

    # ── b. corridor junction_ids from the trained set (same as a real plan) ──
    meta = ModelLoader.get_meta()
    total = _haversine_km(a, b, c, d)
    cands = []
    for _, row in meta.iterrows():
        p = (float(row["latitude"]), float(row["longitude"]))
        d_o = _haversine_km(a, b, p[0], p[1]); d_d = _haversine_km(c, d, p[0], p[1])
        frac = (d_o**2 - d_d**2 + total**2) / (2 * total**2) if total > 0 else 0
        perp = math.sqrt(max(0.0, d_o**2 - (frac * total)**2))
        if 0.0 <= frac <= 1.0 and perp < 2.5:
            cands.append((frac, str(row["junction_id"])))
    cands.sort()
    jids = []
    if cands:
        n = min(10, len(cands))
        step = (len(cands) - 1) / max(1, n - 1)
        for i in range(n):
            jid = cands[int(i * step)][1]
            if jid not in jids:
                jids.append(jid)

    # ── c. create the temporary plan (exactly like a real user plan) ──
    now = datetime.now(timezone.utc)
    trip_row = {
        "user_id": "demo-broadcast",
        "dest_name": f"Demo — {name}",
        "origin_lat": a, "origin_lng": b, "dest_lat": c, "dest_lng": d,
        "leave_by": now.isoformat(),
        "arrive_by": (now + timedelta(seconds=base_dur)).isoformat(),
        "base_duration_seconds": base_dur,
        "predicted_minutes": round(base_dur / 60),
        "junction_ids": jids,
        "status": "active",
    }
    ins = supabase.table("planned_trips").insert(trip_row)
    if not ins.data:
        return {"status": "error", "reason": f"plan insert failed: {ins.error}"}
    trip = ins.data[0]
    trip_id = trip["id"]

    # ── d. run the EXISTING scan function (writes scan_results, decides alert) ──
    await scan_trip(trip, {}, supabase)

    # read the scan's own result for an honest notification level + delay
    sr = (supabase.table("scan_results").select("*")
          .eq("trip_id", trip_id).eq("prediction_source", "google_routes")
          .order("scanned_at", desc=True).limit(1).execute().data or [])
    if sr:
        predicted_jam = float(sr[0].get("predicted_jam_factor") or 0.0)
        level         = sr[0].get("predicted_level") or "FREE"
        live_sec      = float(sr[0].get("user_eta_seconds") or cong["live_seconds"])
    else:
        predicted_jam = round(cong["jam_factor"] * 10, 2)
        level         = _jam_to_level(predicted_jam)
        live_sec      = float(cong["live_seconds"])
    real_delay_min = max(0, round((live_sec - base_dur) / 60))
    eta_minutes    = int(live_sec / 60)
    rep_junction   = jids[0] if jids else "demo"
    # For the demo, always broadcast at least a WARNING-styled card so every
    # device receives a "tap to view" — the alert screen shows the real data.
    notif_level = level if level in ("CRITICAL", "SERIOUS", "WARNING") else "WARNING"

    # ── e. broadcast the EXISTING notification to ALL devices (one call per user) ──
    tok_rows = supabase.table("fcm_tokens").select("user_id").limit(500).execute().data or []
    user_ids = list({r["user_id"] for r in tok_rows if r.get("user_id")})
    notified = 0
    for uid in user_ids:
        try:
            await _send_trip_alert_notification(
                user_id=uid, trip_id=trip_id, dest_name=trip_row["dest_name"],
                junction_id=rep_junction, level=notif_level, kind="new",
                jam_factor=predicted_jam, eta_minutes=eta_minutes,
                delay_min=real_delay_min, prev_delay_min=None,
                leave_by=now, worst_lead_min=None, supabase=supabase)
            notified += 1
        except Exception as e:
            logger.warning(f"demo broadcast failed for user {uid}: {e}")

    logger.info(f"DEMO scan: route={name} level={level} delay={real_delay_min}min "
                f"trip={trip_id} users_notified={notified}")
    return {
        "status": "ok", "trip_id": trip_id, "route": name,
        "congestion_ratio": cong["congestion_ratio"], "level": level,
        "notification_level": notif_level, "real_delay_min": real_delay_min,
        "junctions": len(jids), "users_notified": notified,
    }


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": ModelLoader.is_loaded(),
        "version":      "1.0.0",
    }