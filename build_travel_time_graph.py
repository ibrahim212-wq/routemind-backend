"""
build_travel_time_graph.py

يبني travel-time graph للـ 2076 junction (real + virtual).

Input:
  - artifacts/junction_meta.csv     (460 real junctions)
  - Supabase virtual_junctions table (1616 virtual junctions)

Output:
  - artifacts/combined_meta.csv       (2076 junctions)
  - artifacts/travel_time_graph.pt    (PyTorch graph بـ travel_time)

شغّله مرة واحدة. بعدين deploy الـ backend.
"""

import math
import sys
import requests
import pandas as pd
import numpy as np
import torch

# ─── Config ──────────────────────────────────────────────────────────

ARTIFACTS_DIR  = "artifacts"
SUPABASE_URL   = "https://mldsninoxerbmecojztr.supabase.co"
SUPABASE_KEY   = "sb_secret_hVODQk0eQws67YCEuFo7gg_toLomXCa"
K_NEIGHBORS    = 15   # أقرب 15 junction لكل node

# سرعة متوسطة حسب نوع الطريق (km/h)
ROAD_SPEEDS = {
    "Ring Road":              90,
    "Regional Ring Road":    110,
    "Autostrade":             90,
    "Cairo Alex Desert":     110,
    "Cairo Alex Agricultural": 80,
    "Cairo Suez":            110,
    "Cairo Ismailia":        110,
    "Cairo Fayoum":           90,
    "Cairo Asyut Western":   110,
    "Ain Sokhna Road":       110,
    "Alexandria Coastal":    110,
    "Wahat Road":             90,
    "New Capital Road":      100,
    "Madinaty Road":          80,
    "Shorouk Mostakbal":      70,
    "Mohamed Bin Zayed Axis": 80,
    "July 26 Axis":           70,
    "October Bridge":         70,
    "May 15 Bridge":          70,
}

TIER_SPEEDS = {
    1: 35,   # real junction — main road
    2: 25,   # real junction — secondary
}
DEFAULT_VIRTUAL_SPEED = 60  # virtual junction غير مصنّف


# ─── Helpers ─────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(max(0, min(1, a))))


def get_junction_speed(row):
    """يرجع السرعة المتوسطة بناءً على نوع الـ junction."""
    jtype = row.get("junction_type", "real")
    if jtype == "virtual":
        road_name = row.get("road_name", "")
        return ROAD_SPEEDS.get(road_name, DEFAULT_VIRTUAL_SPEED)
    else:
        tier = int(row.get("tier", 2))
        return TIER_SPEEDS.get(tier, 28)


def travel_time_sec(src_row, tgt_row):
    """يحسب travel_time بالثواني بين junction اثنين."""
    dist_km = haversine_km(
        src_row["latitude"],  src_row["longitude"],
        tgt_row["latitude"], tgt_row["longitude"],
    )
    spd = (get_junction_speed(src_row) + get_junction_speed(tgt_row)) / 2.0
    return (dist_km / max(spd, 1)) * 3600


# ─── Fetch virtual junctions from Supabase ───────────────────────────

def fetch_virtual_junctions():
    print("  📡 جلب virtual junctions من Supabase...")
    all_rows = []
    page = 0
    page_size = 1000

    while True:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/virtual_junctions",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Range":         f"{page*page_size}-{(page+1)*page_size-1}",
            },
            params={
                "select":    "junction_id,lat,lon,road_name,tier",
                "is_active": "eq.true",
            },
            timeout=15,
        )
        if resp.status_code not in (200, 206):
            print(f"  ❌ HTTP {resp.status_code}")
            break

        rows = resp.json()
        if not rows:
            break

        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1

    print(f"  ✅ {len(all_rows)} virtual junction")
    return all_rows


# ─── Build KD-Tree for fast nearest neighbor search ──────────────────

def build_kdtree(lats, lons):
    """
    KD-Tree بـ radians للبحث الجغرافي السريع.
    يرجع tree + coords_rad للـ query.
    """
    try:
        from scipy.spatial import cKDTree
        coords_rad = np.column_stack([
            np.radians(lats),
            np.radians(lons),
        ])
        return cKDTree(coords_rad), coords_rad
    except ImportError:
        return None, None


# ─── Brute-force nearest (fallback لو scipy مش موجود) ──────────────

def find_k_nearest_brute(idx, lats, lons, k):
    """يلاقي الـ K أقرب junctions لـ junction[idx]."""
    distances = []
    for j in range(len(lats)):
        if j == idx:
            continue
        d = haversine_km(lats[idx], lons[idx], lats[j], lons[j])
        distances.append((d, j))
    distances.sort()
    return [j for _, j in distances[:k]]


# ─── Main ────────────────────────────────────────────────────────────

def main():
    # ── 1. Load real junctions ───────────────────────────────────────
    print("=" * 60)
    print("📂 تحميل البيانات")
    print("=" * 60)

    meta_path = f"{ARTIFACTS_DIR}/junction_meta.csv"
    real_df = pd.read_csv(meta_path)
    real_df["junction_type"] = "real"
    print(f"  ✅ Real junctions: {len(real_df)}")

    # ── 2. Fetch virtual junctions ───────────────────────────────────
    virtual_rows = fetch_virtual_junctions()
    if not virtual_rows:
        print("  ⚠️  مفيش virtual junctions — هنشتغل على الـ real بس")
        virtual_df = pd.DataFrame()
    else:
        virtual_df = pd.DataFrame(virtual_rows)
        virtual_df = virtual_df.rename(columns={"lat": "latitude", "lon": "longitude"})
        virtual_df["junction_type"] = "virtual"
        virtual_df["ways_count"]    = 2
        virtual_df["governorate"]   = "N/A"

    # ── 3. Combine ───────────────────────────────────────────────────
    if not virtual_df.empty:
        start_idx = len(real_df)
        virtual_df["junction_idx"] = range(start_idx, start_idx + len(virtual_df))

        # نبني الـ virtual بنفس columns بتاعت الـ real
        v_clean = pd.DataFrame({
            "junction_id":   virtual_df["junction_id"].values,
            "latitude":      virtual_df["latitude"].values,
            "longitude":     virtual_df["longitude"].values,
            "governorate":   "N/A",
            "ways_count":    2,
            "tier":          virtual_df["tier"].values,
            "junction_idx":  virtual_df["junction_idx"].values,
            "junction_type": "virtual",
            "road_name":     virtual_df["road_name"].values,
        })

        real_df["junction_type"] = "real"
        real_df["road_name"]     = ""

        combined = pd.concat([real_df, v_clean], ignore_index=True)
    else:
        real_df["junction_type"] = "real"
        real_df["road_name"]     = ""
        combined = real_df

    print(f"  ✅ Combined: {len(combined)} junctions total")

    # حفظ combined_meta
    combined.to_csv(f"{ARTIFACTS_DIR}/combined_meta.csv", index=False)
    print(f"  💾 Saved: artifacts/combined_meta.csv")

    # ── 4. Build Graph ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🕸️  بناء الـ Travel-Time Graph")
    print("=" * 60)

    N     = len(combined)
    lats  = combined["latitude"].values.astype(float)
    lons  = combined["longitude"].values.astype(float)
    rows_dict = combined.to_dict("records")

    # حاول تستخدم KD-Tree
    tree, coords_rad = build_kdtree(lats, lons)
    use_kdtree = tree is not None
    if use_kdtree:
        print(f"  ✅ KD-Tree جاهز (سريع)")
    else:
        print(f"  ⚠️  scipy مش موجود — هنستخدم brute-force (بطيء)")

    # ── 5. Generate edges ────────────────────────────────────────────
    src_list = []
    tgt_list = []
    tt_list  = []

    print(f"  📊 حساب edges لـ {N} junction × {K_NEIGHBORS} neighbors...")

    for i in range(N):
        if i % 200 == 0:
            pct = (i / N) * 100
            print(f"  ⏳ {i}/{N} ({pct:.0f}%)", end="\r")

        # إيجاد الـ K أقرب
        if use_kdtree:
            # +1 عشان الـ junction نفسه بييجي في النتايج
            dists, indices = tree.query(coords_rad[i], k=K_NEIGHBORS+1)
            neighbors = [j for j in indices if j != i][:K_NEIGHBORS]
        else:
            neighbors = find_k_nearest_brute(i, lats, lons, K_NEIGHBORS)

        # حساب travel_time لكل neighbor
        for j in neighbors:
            tt = travel_time_sec(rows_dict[i], rows_dict[j])

            # نضيف edge في الاتجاهين
            src_list.append(i)
            tgt_list.append(j)
            tt_list.append(tt)

            src_list.append(j)
            tgt_list.append(i)
            tt_list.append(tt)

    print(f"\n  ✅ {len(src_list)} edges")

    # ── 6. Save as PyTorch ───────────────────────────────────────────
    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_weight = torch.tensor(tt_list, dtype=torch.float32)

    # نحفظ كمان node_ids للـ lookup
    node_ids   = combined["junction_id"].tolist()
    node_types = combined["junction_type"].tolist()

    torch.save({
        "edge_index":  edge_index,
        "edge_weight": edge_weight,   # travel_time بالثواني (مش correlation)
        "node_ids":    node_ids,
        "node_types":  node_types,
        "n_nodes":     N,
        "k_neighbors": K_NEIGHBORS,
    }, f"{ARTIFACTS_DIR}/travel_time_graph.pt")

    print(f"  💾 Saved: artifacts/travel_time_graph.pt")

    # ── 7. Stats ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 Stats")
    print("=" * 60)

    tt_arr = np.array(tt_list)
    print(f"  Nodes:          {N:,}")
    print(f"  Edges:          {len(src_list):,}")
    print(f"  Travel time:")
    print(f"    Min:          {tt_arr.min()/60:.1f} min")
    print(f"    Avg:          {tt_arr.mean()/60:.1f} min")
    print(f"    Max:          {tt_arr.max()/60:.1f} min")

    print("\n🎉 Graph built!")
    print("الخطوة الجاية: deploy الـ backend علشان يستخدم")
    print("  travel_time_graph.pt بدل junction_graph.pt")


if __name__ == "__main__":
    main()