"""
model/tier2.py
LSTM inference (Tier 2) — for short-horizon predictions (15-60 min).
Uses current TomTom readings as input.
"""

import numpy as np
import torch
from datetime import datetime
import math
from model.loader import ModelLoader


def _build_feature_vector(reading: dict, dt: datetime, cfg) -> list[float]:
    """
    Build the 16-feature vector for one junction at one timestep.
    Same order as training feature_cols.
    """
    hour = dt.hour
    dow  = dt.weekday()
    quarter = hour * 4 + dt.minute // 15

    # Prophet features for this junction
    pf = ModelLoader.get_prophet_feats()
    jid = reading.get("junction_id", "")
    if jid in pf:
        daily  = float(pf[jid].get("daily",  [0]*96)[quarter])
        weekly = float(pf[jid].get("weekly", [0]*7)[dow])
    else:
        daily = weekly = 0.0

    jam = float(reading.get("jam_factor", 0.0)) / cfg.JAM_SCALE

    return [
        jam,                                            # 1. jam_factor (normalized)
        float(reading.get("current_speed", 30.0)),     # 2. current_speed
        float(reading.get("congestion_ratio", 1.0)),   # 3. congestion_ratio
        float(reading.get("speed_reduction", 0.0)),    # 4. speed_reduction
        float(reading.get("delay_seconds", 0.0)),      # 5. delay_seconds
        float(reading.get("confidence", 0.8)),         # 6. confidence
        math.sin(2 * math.pi * hour / 24),             # 7. hour_sin
        math.cos(2 * math.pi * hour / 24),             # 8. hour_cos
        math.sin(2 * math.pi * dow / 7),               # 9. dow_sin
        math.cos(2 * math.pi * dow / 7),               # 10. dow_cos
        float(dow in [4, 5]),                          # 11. is_weekend
        float(dow == 4),                               # 12. is_friday
        float(reading.get("ways_count", 3)),           # 13. ways_count
        float(reading.get("tier", 1)),                 # 14. tier
        daily,                                         # 15. prophet_daily
        weekly,                                        # 16. prophet_weekly
    ]


def predict_junction_short_horizon(
    junction_id: str,
    history_readings: list[dict],
    history_timestamps: list[datetime],
) -> dict | None:
    """
    Runs LSTM inference for a single junction.

    Args:
        junction_id:        The junction to predict
        history_readings:   Last 12 TomTom readings for ALL junctions
                            Each reading: {junction_id: {...features...}}
        history_timestamps: Timestamps for each reading (len=12)

    Returns:
        {
          "jam_15min": float,  "jam_30min": float,  "jam_60min": float,
          "is_congested": bool,
          "congestion_probability": float,
          "level": str,
        }
        or None if insufficient history
    """
    cfg  = ModelLoader.get_cfg()
    meta = ModelLoader.get_meta()

    if len(history_readings) < cfg.SEQUENCE_LENGTH:
        return None  # Not enough history

    # Find junction index
    junc_row = meta[meta["junction_id"] == junction_id]
    if junc_row.empty:
        return None
    j_idx = int(junc_row.iloc[0]["junction_idx"])
    N     = len(meta)
    T     = cfg.SEQUENCE_LENGTH
    F     = cfg.NODE_FEATURE_DIM

    # Build [T, N, F] feature tensor
    data = np.zeros((T, N, F), dtype=np.float32)

    for t, (readings_snap, dt) in enumerate(
        zip(history_readings[-T:], history_timestamps[-T:])
    ):
        for _, row in meta.iterrows():
            jid = row["junction_id"]
            j   = int(row["junction_idx"])
            if jid in readings_snap:
                feat = _build_feature_vector(readings_snap[jid], dt, cfg)
                data[t, j, :] = feat

    # Scale features
    scaler = ModelLoader.get_scaler()
    scaled = scaler.transform(data.reshape(-1, F)).reshape(T, N, F)

    # Inference
    model       = ModelLoader.get_model()
    edge_index, edge_weight = ModelLoader.get_graph()

    x_seq  = torch.from_numpy(scaled).unsqueeze(0)  # [1, T, N, F]
    j_tens = torch.tensor([j_idx], dtype=torch.long)

    with torch.no_grad():
        reg_p, cls_p = model(x_seq, edge_index, edge_weight, j_tens)

    jam_15 = float(reg_p[0, 0])
    jam_30 = float(reg_p[0, 1])
    jam_60 = float(reg_p[0, 2])

    cls_probs = torch.softmax(cls_p[0], dim=0)
    cong_prob = float(cls_probs[1])
    is_cong   = cong_prob >= 0.5

    def level(j):
        if j < 0.2: return "FREE"
        if j < 0.4: return "LIGHT"
        if j < 0.6: return "MODERATE"
        return "HIGH"

    return {
        "jam_15min": round(jam_15, 3),
        "jam_30min": round(jam_30, 3),
        "jam_60min": round(jam_60, 3),
        "level_15min": level(jam_15),
        "is_congested": is_cong,
        "congestion_probability": round(cong_prob, 3),
    }