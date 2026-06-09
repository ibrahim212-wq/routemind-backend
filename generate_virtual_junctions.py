"""
generate_virtual_junctions.py

يجيب الـ geometry الحقيقية للطرق من Mapbox Directions API
ويولّد virtual junctions كل 2 كم
ويرفعهم لـ Supabase.
"""

import math
import time
import requests

SUPABASE_URL = "https://mldsninoxerbmecojztr.supabase.co"
SUPABASE_KEY = "sb_secret_hVODQk0eQws67YCEuFo7gg_toLomXCa"
MAPBOX_TOKEN = "pk.eyJ1IjoiaWJyYWhpbTIxMiIsImEiOiJjbW9kN3IzeXEwMDV5MnFyMXA4N3l2cjU4In0.zJ4Q_8cHMArhMMwYMI89Ew"

SUPABASE_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}

INTERVAL_KM = 2.0

# ─── الطرق: كل طريق عنده waypoints (lon, lat) بترتيب المسار ──────────
# Mapbox بياخد (longitude, latitude) مش (lat, lon)

ROADS = {

    # ══════ القاهرة والجيزة ══════════════════════════════════

    "Ring_Road": [
        # الدائري - نقاط على امتداده
        (31.337, 30.167), (31.410, 30.155), (31.470, 30.105),
        (31.445, 30.035), (31.325, 29.965), (31.190, 29.980),
        (31.118, 30.080), (31.150, 30.148), (31.275, 30.172),
        (31.337, 30.167),
    ],

    "Autostrade": [
        (31.280, 30.058), (31.350, 30.065), (31.430, 30.068),
        (31.490, 30.065),
    ],

    "Salah_Salem": [
        (31.285, 30.050), (31.263, 30.068), (31.243, 30.090),
        (31.230, 30.122), (31.228, 30.148),
    ],

    "October_Bridge": [
        (31.190, 30.055), (31.230, 30.060), (31.260, 30.058),
        (31.290, 30.055),
    ],

    "Mohamed_Bin_Zayed_Axis": [
        (30.900, 29.980), (31.050, 30.000), (31.200, 30.020),
        (31.350, 30.040), (31.470, 30.055),
    ],

    "July_26_Axis": [
        (31.050, 30.060), (31.150, 30.068), (31.250, 30.072),
        (31.350, 30.075),
    ],

    "May_15_Bridge": [
        (31.240, 29.985), (31.280, 30.000), (31.310, 30.015),
        (31.330, 30.030),
    ],

    "Regional_Ring_Road": [
        # الطريق الإقليمي (R3)
        (31.550, 30.050), (31.650, 30.080), (31.750, 30.100),
        (31.900, 30.050), (32.000, 29.980), (32.000, 29.800),
        (31.850, 29.700), (31.600, 29.750), (31.400, 29.850),
        (31.300, 29.950),
    ],

    "Wahat_Road": [
        (31.210, 30.010), (31.050, 29.960), (30.880, 29.930),
        (30.700, 29.908), (30.500, 29.890), (30.300, 29.875),
    ],

    # ══════ العاصمة الإدارية ومدينتي ══════════════════════════

    "New_Capital_Road": [
        # طريق العاصمة الإدارية الجديدة
        (31.490, 30.068), (31.600, 30.040), (31.700, 30.010),
        (31.800, 29.990), (31.900, 29.970), (32.000, 29.960),
        (32.100, 29.950), (32.200, 29.950),
    ],

    "Madinaty_Road": [
        (31.490, 30.068), (31.550, 30.075), (31.620, 30.080),
        (31.700, 30.078), (31.780, 30.072),
    ],

    "Shorouk_Mostakbal": [
        (31.550, 30.100), (31.650, 30.110), (31.750, 30.115),
        (31.850, 30.105),
    ],

    # ══════ الطرق الخارجة من القاهرة ══════════════════════════

    "Cairo_Alex_Desert": [
        (31.290, 30.070), (31.130, 30.075), (30.970, 30.065),
        (30.800, 30.045), (30.630, 30.020), (30.460, 29.998),
        (30.290, 29.978), (30.120, 29.960), (29.950, 29.945),
        (29.780, 29.940),
    ],

    "Cairo_Alex_Agricultural": [
        (31.290, 30.090), (31.150, 30.110), (31.000, 30.125),
        (30.850, 30.135), (30.700, 30.140), (30.550, 30.138),
        (30.400, 30.130), (30.250, 30.118), (30.100, 30.100),
    ],

    "Cairo_Suez": [
        (31.480, 30.100), (31.600, 30.105), (31.720, 30.102),
        (31.850, 30.095), (31.980, 30.082), (32.100, 30.065),
        (32.230, 30.045),
    ],

    "Cairo_Ismailia": [
        (31.490, 30.070), (31.620, 30.090), (31.760, 30.102),
        (31.900, 30.108), (32.040, 30.108), (32.180, 30.102),
        (32.300, 30.090),
    ],

    "Cairo_Fayoum": [
        (31.210, 30.010), (31.130, 29.950), (31.050, 29.890),
        (30.970, 29.830), (30.890, 29.770), (30.810, 29.710),
        (30.730, 29.650),
    ],

    "Cairo_Asyut_Western": [
        (30.880, 29.930), (30.700, 29.870), (30.520, 29.790),
        (30.340, 29.710), (30.160, 29.640), (29.980, 29.580),
    ],

    "Ain_Sokhna_Road": [
        (31.490, 29.970), (31.620, 29.900), (31.750, 29.830),
        (31.880, 29.760), (32.010, 29.690), (32.140, 29.630),
    ],

    "Alexandria_Coastal": [
        (29.950, 31.200), (29.980, 30.900), (30.010, 30.600),
        (30.020, 30.300), (30.000, 30.000), (29.980, 29.800),
    ],
}


# ─── Geo helpers ─────────────────────────────────────────────────────

def haversine_km(p1, p2):
    R = 6371.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(max(0, min(1, a))))


def decode_polyline(encoded):
    """Decode Mapbox encoded polyline."""
    coords = []
    index = 0
    lat = lng = 0
    while index < len(encoded):
        for is_lat in [True, False]:
            b, shift, result = 0, 0, 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            val = ~(result >> 1) if result & 1 else result >> 1
            if is_lat:
                lat += val
            else:
                lng += val
        coords.append((lat / 1e5, lng / 1e5))
    return coords


def sample_polyline(polyline, interval_km):
    if len(polyline) < 2:
        return polyline
    points = [polyline[0]]
    accumulated = 0.0
    for i in range(len(polyline)-1):
        p1, p2 = polyline[i], polyline[i+1]
        seg = haversine_km(p1, p2)
        if seg == 0:
            continue
        dist = interval_km - accumulated
        while dist <= seg:
            frac = dist / seg
            points.append((p1[0]+frac*(p2[0]-p1[0]), p1[1]+frac*(p2[1]-p1[1])))
            dist += interval_km
        accumulated = seg - (dist - interval_km)
    return points


def deduplicate(points, min_km=1.5):
    if not points:
        return []
    result = [points[0]]
    for p in points[1:]:
        if haversine_km(result[-1], p) >= min_km:
            result.append(p)
    return result


# ─── Mapbox Directions ───────────────────────────────────────────────

def fetch_road_geometry(road_name, waypoints):
    """
    يجيب الـ geometry من Mapbox Directions API.
    waypoints = [(lon, lat), ...] بترتيب المسار
    """
    if len(waypoints) < 2:
        return []

    # Mapbox بياخد max 25 waypoint per request
    # لو عندنا أكتر، نقسمهم
    all_coords = []
    step = 24  # 25 - 1 للـ overlap

    for i in range(0, len(waypoints)-1, step):
        chunk = waypoints[i : i+step+1]
        coords_str = ";".join(f"{lon},{lat}" for lon, lat in chunk)
        url = (
            f"https://api.mapbox.com/directions/v5/mapbox/driving/{coords_str}"
            f"?geometries=polyline&overview=full&access_token={MAPBOX_TOKEN}"
        )
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            routes = data.get("routes", [])
            if not routes:
                continue
            encoded = routes[0]["geometry"]
            coords = decode_polyline(encoded)
            all_coords.extend(coords)
            time.sleep(0.3)
        except Exception as e:
            print(f"\n    ⚠️ Error: {e}")
            continue

    return all_coords


# ─── Main ────────────────────────────────────────────────────────────

def main():
    all_junctions = []
    road_stats    = {}
    empty_roads   = []

    print("=" * 60)
    print("📡 جلب الطرق من Mapbox Directions API")
    print("=" * 60)

    for road_name, waypoints in ROADS.items():
        print(f"  🗺️  {road_name}...", end=" ")

        raw = fetch_road_geometry(road_name, waypoints)
        if not raw:
            empty_roads.append(road_name)
            print("❌ مفيش geometry")
            continue

        sampled  = sample_polyline(raw, INTERVAL_KM)
        filtered = deduplicate(sampled, min_km=1.5)

        for i, (lat, lon) in enumerate(filtered):
            all_junctions.append({
                "junction_id": f"V_{road_name}_{i+1:03d}",
                "lat":         round(lat, 6),
                "lon":         round(lon, 6),
                "road_name":   road_name.replace("_", " "),
                "tier":        1,
                "is_active":   True,
            })

        road_stats[road_name] = len(filtered)
        print(f"✅ {len(filtered)} junction")

    print(f"\n📍 إجمالي: {len(all_junctions)} virtual junction")

    if not all_junctions:
        print("❌ مفيش جداول اترفعوا")
        return

    # ─── Upload ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📤 رفع على Supabase")
    print("=" * 60)

    uploaded = 0
    for i in range(0, len(all_junctions), 100):
        batch = all_junctions[i:i+100]
        resp  = requests.post(
            f"{SUPABASE_URL}/rest/v1/virtual_junctions",
            headers=SUPABASE_HEADERS,
            json=batch,
        )
        if resp.status_code in (200, 201, 204):
            uploaded += len(batch)
            print(f"  ✅ {uploaded}/{len(all_junctions)}", end="\r")
        else:
            print(f"\n  ❌ {resp.status_code}: {resp.text[:150]}")

    print(f"\n\n🎉 خلص! تم رفع {uploaded} virtual junction")
    print("\nالطرق:")
    for road, count in road_stats.items():
        print(f"  ✅ {road.replace('_', ' ')}: {count} junction")
    if empty_roads:
        for r in empty_roads:
            print(f"  ❌ {r.replace('_', ' ')}: فشل")


if __name__ == "__main__":
    main()