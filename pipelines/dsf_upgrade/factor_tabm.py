#!/usr/bin/env python3
"""Small parameter-efficient TabM-like network with explicit ID interactions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from pipelines.mlp.train_walkforward import BASE_CAT, EXTRA_CAT, enrich

PAIRS = [
    ("pitcher_id", "batter_id"), ("pitcher_id", "count_state"),
    ("pitcher_id", "game_type"), ("pitcher_id", "batter_hand"),
    ("pitcher_team_id", "batter_team_id"), ("batter_id", "pitcher_hand"),
]


class Preprocessor:
    def fit(self, x: pd.DataFrame):
        self.cat_cols = [c for c in BASE_CAT + EXTRA_CAT if c in x]
        self.num_cols = [c for c in x if c not in self.cat_cols]
        self.maps, self.card = {}, []
        for c in self.cat_cols:
            values = sorted(x[c].fillna("__MISSING__").astype(str).unique().tolist())
            self.maps[c] = {v: i + 1 for i, v in enumerate(values)}
            self.card.append(len(values) + 1)
        numeric = x[self.num_cols].apply(pd.to_numeric, errors="coerce")
        self.median = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.median)
        self.mean = filled.mean(); self.std = filled.std().replace(0.0, 1.0).fillna(1.0)
        self.pair_indices = [(self.cat_cols.index(a), self.cat_cols.index(b)) for a, b in PAIRS]
        return self

    def transform(self, x: pd.DataFrame):
        cats = np.column_stack([x[c].fillna("__MISSING__").astype(str).map(self.maps[c]).fillna(0).to_numpy(np.int64) for c in self.cat_cols])
        numeric = x[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.median)
        numeric = ((numeric - self.mean) / self.std).clip(-8, 8).to_numpy(np.float32)
        return cats, numeric

    def artifact(self):
        return {"cat_cols": self.cat_cols, "num_cols": self.num_cols, "maps": self.maps,
            "cardinalities": self.card, "median": self.median.to_dict(), "mean": self.mean.to_dict(),
            "std": self.std.to_dict(), "pair_indices": self.pair_indices}


class FactorTabM(nn.Module):
    def __init__(self, cardinalities, n_num, pair_indices, k=8):
        super().__init__()
        dims = [min(24, max(4, int(round(1.6 * c ** 0.25)))) for c in cardinalities]
        self.emb = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, dims)])
        self.inter = nn.ModuleList([nn.Embedding(c, 12) for c in cardinalities])
        self.pairs = pair_indices; self.k = k
        self.input = nn.Linear(sum(dims) + n_num, 192)
        self.norm1 = nn.LayerNorm(192); self.shared = nn.Linear(192, 96); self.norm2 = nn.LayerNorm(96)
        self.r1 = nn.Parameter(torch.ones(k, 192)); self.r2 = nn.Parameter(torch.ones(k, 96))
        self.out = nn.Linear(96, 1); self.head_bias = nn.Parameter(torch.zeros(k))
        self.factor_scale = nn.Parameter(torch.full((k,), 0.05))
        nn.init.normal_(self.r1, 1.0, 0.03); nn.init.normal_(self.r2, 1.0, 0.03)

    def forward(self, cat, num):
        base = torch.cat([e(cat[:, i]) for i, e in enumerate(self.emb)] + [num], dim=1)
        h = torch.nn.functional.silu(self.norm1(self.input(base)))
        h = h[:, None, :] * self.r1[None, :, :]
        h = torch.nn.functional.silu(self.norm2(self.shared(h))) * self.r2[None, :, :]
        logits = self.out(h).squeeze(-1) + self.head_bias
        interaction = 0.0
        for a, b in self.pairs:
            interaction = interaction + (self.inter[a](cat[:, a]) * self.inter[b](cat[:, b])).sum(dim=1) / 12.0
        return logits + interaction[:, None] * self.factor_scale[None, :]


def predict(model, cats, numeric, batch=32768):
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(cats), batch):
            logits = model(torch.as_tensor(cats[start:start + batch]), torch.as_tensor(numeric[start:start + batch]))
            output.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
    return np.concatenate(output)


def make_features(df):
    return enrich(df)
