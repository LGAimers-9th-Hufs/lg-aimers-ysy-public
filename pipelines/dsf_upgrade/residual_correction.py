#!/usr/bin/env python3
"""Walk-forward-stable linear correction around the frozen DSF probability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "control_success"
NUMERIC = [
    "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before",
    "outs_before", "score_diff_pitcher_team", "num_runners_on", "li",
    "home_win_expectancy", "away_win_expectancy", "asof_pitcher_n",
    "asof_batter_n", "asof_pitcher_pitchmix_n", "asof_pitcher_success_rate",
    "asof_batter_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
]
CATEGORICAL = ["game_type", "top_bottom", "base_state", "pitcher_hand", "batter_hand"]
LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1)
SHRINKS = (0.25, 0.5, 0.75, 1.0)
CAP = 0.015


def design(df: pd.DataFrame, pred: np.ndarray, spec: dict | None = None):
    raw: dict[str, np.ndarray] = {
        "dsf_centered": np.asarray(pred, dtype=np.float64) - 0.5,
        "dsf_uncertainty": np.asarray(pred, dtype=np.float64) * (1.0 - np.asarray(pred, dtype=np.float64)),
    }
    for col in NUMERIC:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        if col in {"asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"}:
            values = np.log1p(np.maximum(values, 0.0))
        elif col == "li":
            values = np.log1p(np.maximum(values, 0.0))
        raw[col] = values
    balls = pd.to_numeric(df["balls_before"], errors="coerce").fillna(0).to_numpy(int)
    strikes = pd.to_numeric(df["strikes_before"], errors="coerce").fillna(0).to_numpy(int)
    raw["full_count"] = ((balls == 3) & (strikes == 2)).astype(float)
    raw["count_pressure"] = (balls - strikes).astype(float)
    raw["prev1_minus_long"] = raw["asof_pitcher_prev1_game_success_rate"] - raw["asof_pitcher_success_rate"]
    raw["prev5_minus_long"] = raw["asof_pitcher_prev5_game_success_rate"] - raw["asof_pitcher_success_rate"]
    raw["pitcher_minus_batter"] = raw["asof_pitcher_success_rate"] - raw["asof_batter_success_rate"]
    frame = pd.DataFrame(raw, index=df.index)
    for col in CATEGORICAL:
        values = df[col].astype("string").fillna("<NA>")
        levels = sorted(values.unique().tolist()) if spec is None else spec["levels"][col]
        for level in levels[1:]:
            frame[f"{col}={level}"] = (values == level).to_numpy(float)
    if spec is None:
        center = frame.median().fillna(0.0)
        filled = frame.fillna(center)
        scale = filled.std(ddof=0).replace(0.0, 1.0)
        spec = {
            "columns": list(frame.columns),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "levels": {c: sorted(df[c].astype("string").fillna("<NA>").unique().tolist()) for c in CATEGORICAL},
        }
    frame = frame.reindex(columns=spec["columns"], fill_value=0.0)
    center = np.asarray(spec["center"], dtype=np.float64)
    scale = np.asarray(spec["scale"], dtype=np.float64)
    x = (frame.to_numpy(np.float64) - center) / scale
    x[~np.isfinite(x)] = 0.0
    return x, spec


def fit_ridge(x: np.ndarray, residual: np.ndarray, lam: float):
    mean = float(np.mean(residual))
    centered = residual - mean
    gram = x.T @ x / len(x)
    rhs = x.T @ centered / len(x)
    coef = np.linalg.solve(gram + float(lam) * np.eye(x.shape[1]), rhs)
    return mean, coef


def brier(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def corrected(pred, x, intercept, coef, shrink):
    delta = np.clip(intercept + x @ coef, -CAP, CAP)
    return np.clip(pred + float(shrink) * delta, 0.001, 0.999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--dsf_oof", required=True)
    ap.add_argument("--output", default="./residual_correction.json")
    ap.add_argument("--report", default="./residual_correction_report.json")
    args = ap.parse_args()
    train = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    train["row_id"] = train["row_id"].astype(str)
    with np.load(args.dsf_oof, allow_pickle=False) as z:
        oof = pd.DataFrame({"row_id": z["row_id"].astype(str), "dsf": z["blend_061"], "saved_y": z["y"]})
    data = train.merge(oof, on="row_id", how="inner", validate="one_to_one")
    if not np.array_equal(data[TARGET].to_numpy(float), data.saved_y.to_numpy(float)):
        raise RuntimeError("OOF target mismatch")
    data = data[data.season.isin([2022, 2023, 2024])].reset_index(drop=True)
    pred = data.dsf.to_numpy(float); y = data[TARGET].to_numpy(float); season = data.season.to_numpy(int)
    results = []
    for year in (2023, 2024):
        tr, va = season < year, season == year
        xtr, spec = design(data.loc[tr], pred[tr]); xva, _ = design(data.loc[va], pred[va], spec)
        for lam in LAMBDAS:
            intercept, coef = fit_ridge(xtr, y[tr] - pred[tr], lam)
            for shrink in SHRINKS:
                loss = brier(y[va], corrected(pred[va], xva, intercept, coef, shrink))
                results.append({"year": year, "lambda": lam, "shrink": shrink, "brier": loss, "gain": brier(y[va], pred[va]) - loss})
    candidates = []
    for lam in LAMBDAS:
        for shrink in SHRINKS:
            rows = [r for r in results if r["lambda"] == lam and r["shrink"] == shrink]
            candidates.append({"lambda": lam, "shrink": shrink, "min_gain": min(r["gain"] for r in rows), "mean_gain": np.mean([r["gain"] for r in rows])})
    stable = [c for c in candidates if c["min_gain"] > 0]
    selected = max(stable, key=lambda c: (c["min_gain"], c["mean_gain"])) if stable else None
    report = {"rows": {str(y): int((season == y).sum()) for y in (2022, 2023, 2024)}, "folds": results, "candidates": candidates, "selected": selected, "cap": CAP}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if selected is None:
        print(json.dumps(report, indent=2)); raise SystemExit("no correction improved both walk-forward folds")
    x, spec = design(data, pred)
    intercept, coef = fit_ridge(x, y - pred, selected["lambda"])
    artifact = {"version": 1, "feature_spec": spec, "intercept": intercept, "coef": coef.tolist(), "shrink": selected["shrink"], "cap": CAP, "selection": selected}
    Path(args.output).write_text(json.dumps(artifact), encoding="utf-8")
    print(json.dumps({"selected": selected, "folds": [r for r in results if r["lambda"] == selected["lambda"] and r["shrink"] == selected["shrink"]]}, indent=2))


if __name__ == "__main__":
    main()
