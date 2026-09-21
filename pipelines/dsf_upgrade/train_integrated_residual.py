#!/usr/bin/env python3
"""Strict walk-forward DSF residual model with novel row-independent features."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipelines.dsf_upgrade.novel_features import fit_artifacts, transform
from pipelines.trackman.features import load_artifacts

ID, TARGET = "row_id", "control_success"


def brier(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def bss(y, p):
    y = np.asarray(y, float)
    return 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean())))


def model(seed=7319, rounds=450):
    return lgb.LGBMRegressor(objective="regression_l2", n_estimators=rounds, learning_rate=0.025,
        num_leaves=31, max_depth=6, min_child_samples=700, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=20.0, random_state=seed,
        n_jobs=6, verbosity=-1, deterministic=True, force_col_wise=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--dsf_oof", required=True)
    ap.add_argument("--trackman_artifact", default="./model_trackman/trackman_artifacts.json")
    ap.add_argument("--out_dir", default="./model_dsf_integrated_residual")
    ap.add_argument("--report", default="./dsf_integrated_residual_report.json")
    args = ap.parse_args()
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    frame[ID] = frame[ID].astype(str)
    with np.load(args.dsf_oof, allow_pickle=False) as saved:
        oof = pd.DataFrame({ID: saved["row_id"].astype(str), "dsf": saved["blend_061"].astype(float), "saved_y": saved["y"].astype(float)})
    aligned = frame.merge(oof, on=ID, how="inner", validate="one_to_one")
    if not np.array_equal(aligned[TARGET].to_numpy(float), aligned.saved_y.to_numpy(float)):
        raise RuntimeError("OOF target mismatch")
    tm = load_artifacts(args.trackman_artifact)
    feature_parts = []
    for year in (2022, 2023, 2024):
        rows = aligned[aligned.season == year].copy()
        history = frame[frame.season < year]
        artifacts = fit_artifacts(history, frame)
        x = transform(rows[frame.columns], artifacts, tm)
        x.index = rows.index
        feature_parts.append(x)
        print(f"features year={year} rows={len(rows):,} cols={x.shape[1]}", flush=True)
    features = pd.concat(feature_parts).sort_index()
    if not features.index.equals(aligned.index):
        aligned = aligned.loc[features.index]
    y = aligned[TARGET].to_numpy(float); dsf = aligned.dsf.to_numpy(float); season = aligned.season.to_numpy(int)
    residual = y - dsf
    fold_predictions = np.full(len(aligned), np.nan, dtype=float)
    fold_results = {}
    for year in (2023, 2024):
        train_mask, valid_mask = season < year, season == year
        reg = model(7319 + year)
        reg.fit(features.loc[train_mask], residual[train_mask])
        fold_predictions[valid_mask] = reg.predict(features.loc[valid_mask])
        fold_results[str(year)] = {"train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum())}
    candidates = []
    for cap in (0.003, 0.005, 0.008, 0.012):
        for shrink in (0.1, 0.2, 0.3, 0.5, 0.75):
            record = {"cap": cap, "shrink": shrink, "scores": {}}
            stable = True
            gains = []
            for year in (2023, 2024):
                for regime in ("R", "F"):
                    mask = (season == year) & (aligned.game_type.astype(str).to_numpy() == regime)
                    p = np.clip(dsf[mask] + shrink * np.clip(fold_predictions[mask], -cap, cap), 0.001, 0.999)
                    gain = bss(y[mask], p) - bss(y[mask], dsf[mask])
                    record["scores"][f"{year}_{regime}"] = gain
                    if regime == "R" or year == 2024:
                        gains.append(gain)
                        stable &= gain > 0
            record["min_required_gain"] = float(min(gains)); record["mean_required_gain"] = float(np.mean(gains)); record["stable"] = bool(stable)
            candidates.append(record)
    stable = [c for c in candidates if c["stable"]]
    selected = max(stable, key=lambda c: (c["min_required_gain"], c["mean_required_gain"])) if stable else None
    report = {"folds": fold_results, "candidates": candidates, "selected": selected,
        "rules": "official train/TrackMan and DSF train OOF only; no test or leaderboard fitting"}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected}, ensure_ascii=False, indent=2))
    if selected is None:
        raise SystemExit("no candidate improved required temporal/regime folds")
    final_artifacts = fit_artifacts(frame, frame)
    final_features = transform(aligned[frame.columns], final_artifacts, tm)
    final = model(7319); final.fit(final_features, residual)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    final.booster_.save_model(str(out / "residual.txt"))
    (out / "feature_artifacts.json").write_text(json.dumps(final_artifacts, ensure_ascii=False), encoding="utf-8")
    meta = {"version": 1, "model_file": "residual.txt", "feature_artifact": "feature_artifacts.json",
        "trackman_artifact": "trackman_artifacts.json", "feature_cols": final_features.columns.tolist(),
        "cap": selected["cap"], "shrink": selected["shrink"], "selection": selected,
        "training_rows": len(aligned), "rules": report["rules"]}
    (out / "artifacts.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(args.trackman_artifact, out / "trackman_artifacts.json")


if __name__ == "__main__":
    main()
