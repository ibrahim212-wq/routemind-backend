"""
migrate_historical_data.py

Migration script: cairo_traffic_data.csv → Supabase junction_readings

شغّله مرة واحدة بس. لو اتقطع في النص، يكمل من checkpoint.
"""

import os
import sys
import time
import json
import math
from pathlib import Path
import pandas as pd
import requests

# ─── Config ──────────────────────────────────────────────────────────
CSV_PATH = r"G:\downloads\cairo_traffic_data.csv"

SUPABASE_URL  = "https://mldsninoxerbmecojztr.supabase.co"
SUPABASE_KEY  = "sb_secret_hVODQk0eQws67YCEuFo7gg_toLomXCa"
CHUNK_SIZE       = 500
CHECKPOINT_FILE  = "migration_checkpoint.json"

HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}


# ─── Helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> int:
    """يرجع آخر row اتعمله insert."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f).get("last_row", 0)
    return 0


def save_checkpoint(row: int):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_row": row}, f)


def transform_row(row) -> dict:
    """يحول row من CSV لـ Supabase format."""
    current_speed   = float(row["current_speed"])
    free_flow_speed = float(row["free_flow_speed"])

    if free_flow_speed > 0:
        congestion_ratio = max(0.0, 1.0 - current_speed / free_flow_speed)
    else:
        congestion_ratio = 0.0

    return {
        "junction_id":      str(row["junction_id"]),
        "recorded_at":      str(row["timestamp"]),
        "jam_factor":       float(row["jam_factor"]),
        "current_speed":    round(current_speed, 1),
        "free_flow_speed":  round(free_flow_speed, 1),
        "congestion_ratio": round(congestion_ratio, 3),
        "speed_reduction":  round(float(row["speed_reduction"]), 1),
        "delay_seconds":    int(row["delay_seconds"]),
        "confidence":       round(float(row["confidence"]), 2),
        "hour":             int(row["hour"]),
        "day_of_week":      int(row["day_of_week"]),
        "is_weekend":       bool(row["is_weekend"]),
        "is_friday":        bool(row["is_friday"]),
        "scan_trigger":     "historical_import",
    }


def insert_batch(records: list) -> bool:
    """يبعت batch لـ Supabase. يرجع True لو نجح."""
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/junction_readings",
            headers=HEADERS,
            json=records,
            timeout=30,
        )
        if resp.status_code in (200, 201, 204):
            return True
        print(f"\n❌ Error {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        return False


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print(f"📂 Reading CSV: {CSV_PATH}")
    if not Path(CSV_PATH).exists():
        print(f"❌ CSV مش موجود: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"✅ Loaded {total:,} rows")

    start_row = load_checkpoint()
    if start_row > 0:
        print(f"📍 Resuming from row {start_row:,}")

    inserted = start_row
    failed_chunks = 0
    start_time = time.time()

    for i in range(start_row, total, CHUNK_SIZE):
        chunk = df.iloc[i : i + CHUNK_SIZE]
        records = [transform_row(row) for _, row in chunk.iterrows()]

        success = insert_batch(records)

        if not success:
            failed_chunks += 1
            if failed_chunks >= 3:
                print(f"\n🛑 3 chunks فشلوا متتاليين — توقفنا عند row {i:,}")
                save_checkpoint(i)
                sys.exit(1)
            print("⏳ Retry بعد 5 ثواني...")
            time.sleep(5)
            continue

        failed_chunks = 0
        inserted = min(i + CHUNK_SIZE, total)

        # Progress
        pct = (inserted / total) * 100
        elapsed = time.time() - start_time
        rate = (inserted - start_row) / max(elapsed, 1)
        eta_sec = (total - inserted) / max(rate, 1)
        eta_min = eta_sec / 60

        print(
            f"✅ {inserted:,}/{total:,} ({pct:.1f}%) | "
            f"{rate:.0f} rows/sec | "
            f"ETA: {eta_min:.1f} min",
            end="\r",
        )

        # Save checkpoint كل 5000 row
        if inserted % 5000 < CHUNK_SIZE:
            save_checkpoint(inserted)

    save_checkpoint(total)
    elapsed_min = (time.time() - start_time) / 60
    print(f"\n\n🎉 Migration done!")
    print(f"   Total inserted: {inserted:,} rows")
    print(f"   Time: {elapsed_min:.1f} minutes")
    print(f"\n   احذف {CHECKPOINT_FILE} لو عايز تعمل migration تاني.")


if __name__ == "__main__":
    main()