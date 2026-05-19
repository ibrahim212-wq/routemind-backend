/**
 * Firebase Cloud Functions — RouteMind Trip Monitor
 *
 * Functions:
 *   scanActiveTrips   → runs every 15 min, scans all upcoming trips
 *   sendFinalClear    → runs every 10 min, sends "all clear" 10 min before departure
 *
 * Firestore structure:
 *   planned_trips/{trip_id}:
 *     user_id, junction_ids, departure_time, base_duration_seconds,
 *     status ("active" | "completed" | "cancelled")
 *
 *   junction_history/{junction_id}/readings/{timestamp}:
 *     current_speed, jam_factor, congestion_ratio, ...
 *
 *   fcm_tokens/{user_id}:
 *     token: string
 */

const { onSchedule }   = require("firebase-functions/v2/scheduler");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore }  = require("firebase-admin/firestore");
const { getMessaging }  = require("firebase-admin/messaging");
const axios             = require("axios");

initializeApp();
const db  = getFirestore();
const fcm = getMessaging();

// ── Config ───────────────────────────────────────────────────────────────────
const AI_BACKEND_URL = process.env.AI_BACKEND_URL || "https://your-cloud-run-url.run.app";
const MONITOR_WINDOW_HOURS = 1.0;   // Start monitoring 1 hour before departure
const ALERT_THRESHOLD_MIN  = 5.0;   // Alert if delay > 5 minutes
const FINAL_CLEAR_MIN      = 10;    // Send "all clear" 10 min before departure

// ── Helper: get FCM token for user ───────────────────────────────────────────
async function getUserFcmToken(userId) {
  const doc = await db.collection("fcm_tokens").doc(userId).get();
  return doc.exists ? doc.data().token : null;
}

// ── Helper: send FCM notification ────────────────────────────────────────────
async function sendNotification(token, title, body, data = {}) {
  if (!token) return;
  try {
    await fcm.send({
      token,
      notification: { title, body },
      data: { ...data, click_action: "FLUTTER_NOTIFICATION_CLICK" },
      android: {
        priority: "high",
        notification: { channelId: "routemind_traffic" },
      },
    });
  } catch (err) {
    console.error("FCM error:", err.message);
  }
}

// ── Helper: get last 12 junction readings from Firestore ─────────────────────
async function getJunctionHistory(junctionIds) {
  const history = {};
  const cutoff  = new Date(Date.now() - 3 * 60 * 60 * 1000); // 3 hours ago

  await Promise.all(junctionIds.map(async (jid) => {
    const snap = await db
      .collection("junction_history")
      .doc(jid)
      .collection("readings")
      .where("timestamp", ">=", cutoff.toISOString())
      .orderBy("timestamp", "desc")
      .limit(12)
      .get();

    history[jid] = snap.docs.map(d => d.data()).reverse();
  }));

  return history;
}

// ── FUNCTION 1: Scan active trips every 15 minutes ───────────────────────────
exports.scanActiveTrips = onSchedule("every 15 minutes", async () => {
  const now           = new Date();
  const windowEnd     = new Date(now.getTime() + MONITOR_WINDOW_HOURS * 60 * 60 * 1000);

  // Get all active trips departing within the monitoring window
  const tripsSnap = await db
    .collection("planned_trips")
    .where("status", "==", "active")
    .where("departure_time", ">=", now.toISOString())
    .where("departure_time", "<=", windowEnd.toISOString())
    .get();

  if (tripsSnap.empty) {
    console.log("No active trips in monitoring window.");
    return;
  }

  console.log(`Scanning ${tripsSnap.size} active trips...`);

  await Promise.all(tripsSnap.docs.map(async (doc) => {
    const trip = doc.data();
    const tripId = doc.id;

    try {
      // Get junction history for LSTM
      const junctionHistory = await getJunctionHistory(trip.junction_ids);

      // Call AI Backend
      const response = await axios.post(`${AI_BACKEND_URL}/api/scan-route`, {
        trip_id:               tripId,
        user_id:               trip.user_id,
        junction_ids:          trip.junction_ids,
        departure_time:        trip.departure_time,
        scan_time:             now.toISOString(),
        base_duration_seconds: trip.base_duration_seconds,
        junction_history:      junctionHistory,
      }, { timeout: 10000 });

      const result = response.data;
      console.log(`Trip ${tripId}: ${result.alert_type}, delay=${result.total_delay_min}min`);

      // Save scan result to Firestore
      await doc.ref.collection("scans").add({
        ...result,
        created_at: now.toISOString(),
      });

      // Send notification if needed
      if (result.should_alert && result.total_delay_min >= ALERT_THRESHOLD_MIN) {
        const token = await getUserFcmToken(trip.user_id);
        const title = result.alert_type === "leave_early"
          ? "🚦 اتحرك بدري — RouteMind"
          : "⚠️ زحمة خفيفة — RouteMind";

        await sendNotification(token, title, result.message_ar, {
          trip_id:        tripId,
          alert_type:     result.alert_type,
          delay_minutes:  String(result.total_delay_min),
        });
      }

    } catch (err) {
      console.error(`Error scanning trip ${tripId}:`, err.message);
    }
  }));
});

// ── FUNCTION 2: Send "all clear" 10 minutes before departure ─────────────────
exports.sendFinalClear = onSchedule("every 10 minutes", async () => {
  const now    = new Date();
  const soon   = new Date(now.getTime() + FINAL_CLEAR_MIN * 60 * 1000);
  const sooner = new Date(now.getTime() + (FINAL_CLEAR_MIN - 5) * 60 * 1000);

  // Get trips departing in the next 10-15 minutes
  const tripsSnap = await db
    .collection("planned_trips")
    .where("status", "==", "active")
    .where("departure_time", ">=", sooner.toISOString())
    .where("departure_time", "<=", soon.toISOString())
    .where("final_clear_sent", "==", false)
    .get();

  await Promise.all(tripsSnap.docs.map(async (doc) => {
    const trip = doc.data();

    // Check if any recent scan showed an alert
    const recentScans = await doc.ref
      .collection("scans")
      .where("created_at", ">=", new Date(Date.now() - 20 * 60000).toISOString())
      .orderBy("created_at", "desc")
      .limit(1)
      .get();

    const lastAlert = recentScans.docs[0]?.data()?.should_alert || false;

    if (!lastAlert) {
      // Route is clear — send final confirmation
      const token = await getUserFcmToken(trip.user_id);
      await sendNotification(
        token,
        "✅ كل تمام — RouteMind",
        `الطريق واضح. اتحرك في معادك!`,
        { trip_id: doc.id, alert_type: "all_clear" }
      );

      // Mark as sent
      await doc.ref.update({ final_clear_sent: true });
    }
  }));
});