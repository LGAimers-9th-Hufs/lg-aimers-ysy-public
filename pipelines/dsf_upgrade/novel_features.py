#!/usr/bin/env python3
"""Train-frozen current-season decomposition plus hierarchical/TrackMan features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.lightgbm_v3.train import base_artifact
from pipelines.lightgbm_v2.inference import build_features as build_v3_features
from pipelines.trackman.features import enrich_trackman
from shared.hierarchical_te_features import fit_artifact as fit_hte, transform as transform_hte

RATE_GROUPS = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]),
    "batter": ("batter_id", "asof_batter_n", ["asof_batter_success_rate", "asof_batter_middle_rate"]),
    "pitchmix": ("pitcher_id", "asof_pitcher_pitchmix_n", [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]),
}


def fit_endpoints(history: pd.DataFrame) -> dict:
    result = {}
    for name, (entity, ncol, rates) in RATE_GROUPS.items():
        if not len(history):
            result[name] = {}
            continue
        work = history[[entity, "season", ncol, *rates]].copy()
        work[ncol] = pd.to_numeric(work[ncol], errors="coerce").fillna(0.0)
        work = work.sort_values([entity, "season", ncol])
        endpoint = work.groupby(entity, observed=True, sort=False).tail(1)
        table = {}
        for row in endpoint.itertuples(index=False, name=None):
            table[str(row[0])] = [float(row[2]), *[float(v) if pd.notna(v) else 0.5 for v in row[3:]]]
        result[name] = table
    return result


def transform_endpoints(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, (entity, ncol, rates) in RATE_GROUPS.items():
        keys = df[entity].astype(str)
        table = artifacts.get(name, {})
        previous = np.array([table.get(k, [0.0] + [0.5] * len(rates)) for k in keys], dtype=np.float64)
        total_n = pd.to_numeric(df[ncol], errors="coerce").fillna(0.0).to_numpy(float)
        current_n = np.maximum(total_n - previous[:, 0], 0.0)
        out[f"cs_{name}_logn"] = np.log1p(current_n).astype(np.float32)
        out[f"cs_{name}_share"] = (current_n / np.maximum(total_n, 1.0)).astype(np.float32)
        for j, rate in enumerate(rates, 1):
            total_rate = pd.to_numeric(df[rate], errors="coerce").to_numpy(float)
            total_rate = np.where(np.isfinite(total_rate), total_rate, previous[:, j])
            current_sum = np.clip(total_n * total_rate - previous[:, 0] * previous[:, j], 0.0, current_n)
            current_rate = (current_sum + 60.0 * previous[:, j]) / (current_n + 60.0)
            tag = rate.removeprefix("asof_").replace("_rate", "")
            out[f"cs_{tag}"] = current_rate.astype(np.float32)
            out[f"cs_{tag}_delta"] = (current_rate - previous[:, j]).astype(np.float32)
    return out


def fit_artifacts(history: pd.DataFrame, schema_frame: pd.DataFrame | None = None) -> dict:
    schema = history if schema_frame is None else schema_frame
    return {"v3": base_artifact(schema, history), "hte": fit_hte(history), "endpoints": fit_endpoints(history)}


def transform(df: pd.DataFrame, artifacts: dict, trackman_artifacts: dict) -> pd.DataFrame:
    blocks = [
        build_v3_features(df, artifacts["v3"]).reset_index(drop=True),
        transform_hte(df, artifacts["hte"]).reset_index(drop=True),
        transform_endpoints(df, artifacts["endpoints"]).reset_index(drop=True),
        enrich_trackman(df, trackman_artifacts).reset_index(drop=True),
    ]
    out = pd.concat(blocks, axis=1)
    out.columns = [str(c) for c in out.columns]
    if out.columns.duplicated().any():
        raise RuntimeError("duplicate integrated feature name")
    return out.astype(np.float32)
