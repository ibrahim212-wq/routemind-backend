"""
model/loader.py
Loads all model artifacts once at startup and keeps them in memory.
"""

import os
import pickle
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path

logger = logging.getLogger("routemind.loader")

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    JAM_SCALE           = 10.0
    BIN_THRESH          = 0.2
    SEQUENCE_LENGTH     = 12
    PREDICTION_HORIZONS = [1, 2, 4]   # +15, +30, +60 min
    NODE_FEATURE_DIM    = 16
    GCN_HIDDEN_DIM      = 64
    GCN_OUTPUT_DIM      = 32
    LSTM_HIDDEN_DIM     = 128
    LSTM_LAYERS         = 2
    DECISION_HIDDEN     = 256
    N_CLASSES           = 2
    DROPOUT             = 0.3
    DEVICE              = torch.device("cpu")

    ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))


# ── Model Architecture ────────────────────────────────────────────────────────
class GCNBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hid_dim)
        self.gcn2 = GCNConv(hid_dim, out_dim)
        self.drop = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.drop, training=self.training)
        return F.relu(self.gcn2(h, edge_index, edge_weight))


class RouteMindModel(nn.Module):
    def __init__(self, num_junctions, cfg: Config):
        super().__init__()
        self.N   = num_junctions
        self.gcn = GCNBlock(cfg.NODE_FEATURE_DIM, cfg.GCN_HIDDEN_DIM,
                            cfg.GCN_OUTPUT_DIM, cfg.DROPOUT)
        self.lstm = nn.LSTM(cfg.GCN_OUTPUT_DIM, cfg.LSTM_HIDDEN_DIM,
                            cfg.LSTM_LAYERS, batch_first=True,
                            dropout=cfg.DROPOUT if cfg.LSTM_LAYERS > 1 else 0)
        self.reg_head = nn.Sequential(
            nn.Linear(cfg.LSTM_HIDDEN_DIM, cfg.DECISION_HIDDEN), nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.DECISION_HIDDEN, cfg.DECISION_HIDDEN // 2), nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.DECISION_HIDDEN // 2, len(cfg.PREDICTION_HORIZONS)),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.LSTM_HIDDEN_DIM, 128), nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(128, cfg.N_CLASSES),
        )

    def forward(self, x_seq, edge_index, edge_weight, junction_idx):
        B, T, N, F = x_seq.shape
        BT = B * T
        x_nodes = x_seq.reshape(BT * N, F)
        E = edge_index.shape[1]
        offsets = torch.arange(BT, device=edge_index.device) * N
        ei_bt = (edge_index.unsqueeze(0).expand(BT, -1, -1) +
                 offsets.view(BT, 1, 1)).reshape(2, BT * E)
        ew_bt = edge_weight.unsqueeze(0).expand(BT, -1).reshape(-1)
        gcn_out = self.gcn(x_nodes, ei_bt, ew_bt).reshape(B, T, N, -1)
        junc_exp = junction_idx.view(B, 1, 1, 1).expand(B, T, 1, gcn_out.size(-1))
        junc_emb = gcn_out.gather(2, junc_exp).squeeze(2)
        lstm_out, _ = self.lstm(junc_emb)
        h = lstm_out[:, -1, :]
        return self.reg_head(h), self.cls_head(h)


# ── Singleton Loader ──────────────────────────────────────────────────────────
class ModelLoader:
    _model          = None
    _scaler         = None
    _hist_lookup    = None
    _meta           = None    # 460 real junctions — للـ backward compatibility
    _combined_meta  = None    # 2076 (real + virtual) — للـ upstream discovery
    _edge_index     = None    # travel_time_graph edges
    _edge_weight    = None    # travel_time بالثواني
    _feature_cols   = None
    _prophet_feats  = None
    _cfg            = Config()
    _loaded         = False

    @classmethod
    def load(cls):
        import pandas as pd
        d = cls._cfg.ARTIFACTS_DIR

        # ── 1. Junction metadata (460 real) ──────────────────────────
        cls._meta = pd.read_csv(d / "junction_meta.csv")
        N_real = len(cls._meta)
        logger.info(f"Real junctions loaded: {N_real}")

        # ── 2. Combined metadata (real + virtual) ────────────────────
        combined_path = d / "combined_meta.csv"
        if combined_path.exists():
            cls._combined_meta = pd.read_csv(combined_path)
            logger.info(f"Combined junctions loaded: {len(cls._combined_meta)}")
        else:
            # Fallback: بس الـ real junctions
            cls._combined_meta = cls._meta.copy()
            cls._combined_meta["junction_type"] = "real"
            cls._combined_meta["road_name"]     = ""
            logger.warning("combined_meta.csv مش موجود — هنستخدم real junctions بس")

        # ── 3. Feature columns ────────────────────────────────────────
        with open(d / "feature_cols.json") as f:
            cls._feature_cols = json.load(f)

        # ── 4. Scaler ─────────────────────────────────────────────────
        with open(d / "scaler.pkl", "rb") as f:
            cls._scaler = pickle.load(f)

        # ── 5. Historical lookup (Tier 1) ─────────────────────────────
        with open(d / "historical_lookup.pkl", "rb") as f:
            data = pickle.load(f)
            cls._hist_lookup = data["lookup"]

        # ── 6. Graph ──────────────────────────────────────────────────
        # الأولوية: travel_time_graph.pt (الجديد) → junction_graph.pt (القديم)
        tt_graph_path  = d / "travel_time_graph.pt"
        old_graph_path = d / "junction_graph.pt"

        if tt_graph_path.exists():
            graph = torch.load(tt_graph_path, map_location="cpu")
            cls._edge_index  = graph["edge_index"]
            cls._edge_weight = graph["edge_weight"]   # travel_time بالثواني
            logger.info(
                f"Travel-time graph loaded: "
                f"{graph['n_nodes']} nodes, "
                f"{cls._edge_index.shape[1]} edges"
            )
        else:
            # Fallback: الـ correlation graph القديم
            graph = torch.load(old_graph_path, map_location="cpu")
            cls._edge_index  = graph["edge_index"]
            cls._edge_weight = graph["edge_weight"]
            logger.warning(
                "travel_time_graph.pt مش موجود — "
                "هنستخدم junction_graph.pt (correlation-based)"
            )

        # ── 7. Prophet features ───────────────────────────────────────
        raw = np.load(d / "prophet_features.npz", allow_pickle=True)
        cls._prophet_feats = {k: raw[k].item() for k in raw.files}

        # ── 8. LSTM Model (Tier 2) ────────────────────────────────────
        # الموديل اتتدرب على N_real=460 — ما نغيّرش ده
        cls._model = RouteMindModel(N_real, cls._cfg)
        ckpt = torch.load(d / "routemind_model.pth", map_location="cpu")
        cls._model.load_state_dict(ckpt["model_state_dict"])
        cls._model.eval()

        cls._loaded = True
        logger.info("✅ All artifacts loaded")

    # ── Getters ───────────────────────────────────────────────────────────────

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    @classmethod
    def get_meta(cls):
        """460 real junctions — للـ junction_mapper و tier1."""
        return cls._meta

    @classmethod
    def get_combined_meta(cls):
        """2076 junctions (real + virtual) — للـ upstream_discovery و scanner."""
        return cls._combined_meta

    @classmethod
    def get_hist_lookup(cls):
        return cls._hist_lookup

    @classmethod
    def get_scaler(cls):
        return cls._scaler

    @classmethod
    def get_model(cls):
        return cls._model

    @classmethod
    def get_graph(cls):
        """
        يرجع (edge_index, edge_weight).
        edge_weight = travel_time بالثواني (من travel_time_graph.pt)
        أو correlation (من junction_graph.pt كـ fallback).
        """
        return cls._edge_index, cls._edge_weight

    @classmethod
    def get_feature_cols(cls):
        return cls._feature_cols

    @classmethod
    def get_prophet_feats(cls):
        return cls._prophet_feats

    @classmethod
    def get_cfg(cls):
        return cls._cfg