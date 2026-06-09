"""
services/upstream_discovery.py

Upstream junction discovery for RouteMind Intelligent Scan.

الفكرة:
  لو اليوزر هيوصل J_target بعد T ثانية،
  نلاقي الـ junctions اللي العربيات دلوقتي فيها
  هتوصل J_target بعد T ثانية بالظبط (± tolerance).

  دي العربيات اللي لو اتأخرت دلوقتي، اليوزر هيحس بيها
  لما يوصل الـ junction بتاعه.

Algorithm: Reverse Dijkstra من الـ target لأقرب كل node.
"""

from __future__ import annotations

import heapq
import math
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# سرعة متوسطة حسب tier الـ junction
SPEED_BY_TIER: Dict[int, float] = {
    1: 35.0,   # main roads / arteries
    2: 25.0,   # secondary / inner-city
}
DEFAULT_SPEED_KMH = 28.0


# ─────────────────────────────────────────────────────────────────
# Geo
# ─────────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2-lat1)/2)**2
         + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2)
    return 2 * R * math.asin(math.sqrt(max(0, min(1, a))))


# ─────────────────────────────────────────────────────────────────
# Build reverse adjacency list  (يتعمل مرة واحدة في الـ startup)
# ─────────────────────────────────────────────────────────────────

def build_reverse_adj(
    edge_index: torch.Tensor,
    meta_df: pd.DataFrame,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    يبني reverse adjacency list من الـ graph.

    reverse_adj[tgt_idx] = [(src_idx, travel_time_sec), ...]

    بيستخدم المسافة الجغرافية + سرعة متوسطة حسب الـ tier
    علشان يحسب travel_time لكل edge.

    Returns:
        dict: tgt_idx → [(src_idx, travel_time_sec), ...]
    """
    # نبني lookup سريع: junction_idx → (lat, lon, tier)
    lookup: Dict[int, Tuple[float, float, int]] = {}
    for _, row in meta_df.iterrows():
        lookup[int(row["junction_idx"])] = (
            float(row["latitude"]),
            float(row["longitude"]),
            int(row.get("tier", 2)),
        )

    reverse_adj: Dict[int, List[Tuple[int, float]]] = {}
    num_edges = edge_index.shape[1]

    for i in range(num_edges):
        src = int(edge_index[0][i].item())
        tgt = int(edge_index[1][i].item())

        if src not in lookup or tgt not in lookup:
            continue

        slat, slon, stier = lookup[src]
        tlat, tlon, ttier = lookup[tgt]

        dist_km = _haversine_km(slat, slon, tlat, tlon)

        # سرعة = متوسط بين tier المصدر والهدف
        spd = (SPEED_BY_TIER.get(stier, DEFAULT_SPEED_KMH)
               + SPEED_BY_TIER.get(ttier, DEFAULT_SPEED_KMH)) / 2.0

        travel_time_sec = (dist_km / spd) * 3600.0

        reverse_adj.setdefault(tgt, []).append((src, travel_time_sec))

    logger.info(
        f"Reverse adj built: {num_edges} edges, "
        f"{len(reverse_adj)} nodes with upstream neighbors"
    )
    return reverse_adj


# ─────────────────────────────────────────────────────────────────
# Reverse Dijkstra
# ─────────────────────────────────────────────────────────────────

def _dijkstra_reverse(
    target_idx: int,
    reverse_adj: Dict[int, List[Tuple[int, float]]],
    max_time_sec: float,
) -> Dict[int, float]:
    """
    Dijkstra من الـ target بالعكس.
    بيحسب travel_time من كل junction للـ target.

    Args:
        target_idx:    junction_idx الـ target
        reverse_adj:   output بتاع build_reverse_adj()
        max_time_sec:  نوقف البحث لو تعدى الوقت ده (optimization)

    Returns:
        {junction_idx: travel_time_to_target_sec}
    """
    dist: Dict[int, float] = {target_idx: 0.0}
    heap = [(0.0, target_idx)]

    while heap:
        d, u = heapq.heappop(heap)

        if d > dist.get(u, float("inf")):
            continue  # outdated entry

        if d > max_time_sec:
            continue  # مش محتاجين أبعد من كده

        for v, travel_time in reverse_adj.get(u, []):
            new_d = d + travel_time
            if new_d < dist.get(v, float("inf")):
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v))

    return dist


# ─────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────

def get_upstream_junctions(
    target_junction_id: str,
    time_offset_seconds: float,
    meta_df: pd.DataFrame,
    reverse_adj: Dict[int, List[Tuple[int, float]]],
    tolerance_pct: float = 0.25,
) -> List[Dict]:
    """
    لكل junction على الطريق، نلاقي الـ junctions اللي
    travel_time بتاعتهم للـ target ≈ time_offset_seconds.

    Args:
        target_junction_id:  "J_0050"
        time_offset_seconds: ETA بتاع اليوزر للـ target بالثواني
        meta_df:             junction_meta.csv
        reverse_adj:         output بتاع build_reverse_adj() (cached)
        tolerance_pct:       ± كام % حول الـ time_offset نقبله (default 25%)

    Returns:
        list of {junction_id, junction_idx, latitude, longitude,
                 travel_time_to_target_sec, time_offset_requested_sec}
        مرتبة: الأقرب لـ time_offset الأول
    """
    if time_offset_seconds <= 0:
        logger.warning(f"get_upstream_junctions: time_offset={time_offset_seconds} — invalid")
        return []

    # نلاقي الـ target idx
    row = meta_df[meta_df["junction_id"] == target_junction_id]
    if row.empty:
        logger.warning(f"get_upstream_junctions: {target_junction_id} مش موجود في الـ meta")
        return []

    target_idx = int(row.iloc[0]["junction_idx"])

    # Dijkstra بالعكس من الـ target
    max_search_sec = time_offset_seconds * (1 + tolerance_pct) + 60  # margin
    all_dists = _dijkstra_reverse(target_idx, reverse_adj, max_search_sec)

    # فلتر: الـ nodes اللي travel_time بتاعتهم ≈ time_offset
    tolerance_sec = time_offset_seconds * tolerance_pct
    results = []

    idx_to_row = {int(r["junction_idx"]): r for _, r in meta_df.iterrows()}

    for idx, travel_time in all_dists.items():
        if idx == target_idx:
            continue
        if abs(travel_time - time_offset_seconds) <= tolerance_sec:
            jrow = idx_to_row.get(idx)
            if jrow is None:
                continue
            results.append({
                "junction_id":               str(jrow["junction_id"]),
                "junction_idx":              idx,
                "latitude":                  float(jrow["latitude"]),
                "longitude":                 float(jrow["longitude"]),
                "travel_time_to_target_sec": round(travel_time, 1),
                "time_offset_requested_sec": round(time_offset_seconds, 1),
                "delta_sec":                 round(abs(travel_time - time_offset_seconds), 1),
            })

    # مرتب: الأقرب لـ time_offset الأول
    results.sort(key=lambda x: x["delta_sec"])

    logger.info(
        f"Upstream of {target_junction_id} at T={time_offset_seconds/60:.1f}min: "
        f"{len(results)} junctions found (tolerance=±{tolerance_pct*100:.0f}%)"
    )
    return results