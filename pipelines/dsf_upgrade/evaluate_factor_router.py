#!/usr/bin/env python3
"""Temporal validation for a row-local DSF/FactorTabM conditional router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def brier(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def bss(y, p):
    y = np.asarray(y, float)
    return 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean())))


def router_features(d):
    dsf, factor = d["dsf"].astype(float), d["factor"].astype(float)
    diff = factor - dsf
    return np.column_stack([
        dsf, factor, diff, np.abs(diff),
        np.abs(dsf - 0.5), np.abs(factor - 0.5), dsf * (1.0 - dsf),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dir", default="./factor_tabm_oof")
    ap.add_argument("--out", default="./factor_tabm_oof/router_results.json")
    args = ap.parse_args()
    data = {year: dict(np.load(Path(args.oof_dir) / f"factor_{year}.npz")) for year in (2023, 2024)}
    d23, d24 = data[2023], data[2024]
    x23, x24 = router_features(d23), router_features(d24)
    y23, y24 = d23["y"].astype(float), d24["y"].astype(float)
    better23 = ((y23 - d23["factor"]) ** 2 < (y23 - d23["dsf"]) ** 2).astype(int)
    model = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.025, num_leaves=7, max_depth=3,
        min_child_samples=1500, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=20.0, verbosity=-1, random_state=44123, n_jobs=6,
    )
    model.fit(x23[d23["stop"]], better23[d23["stop"]])
    route23, route24 = model.predict_proba(x23)[:, 1], model.predict_proba(x24)[:, 1]
    choices = []
    for threshold in np.arange(0.50, 0.901, 0.05):
        for alpha in (0.05, 0.10, 0.15, 0.20, 0.30):
            p = d23["dsf"] + (route23 >= threshold) * alpha * (d23["factor"] - d23["dsf"])
            choices.append((brier(y23[d23["select"]], p[d23["select"]]), threshold, alpha))
    _, threshold, alpha = min(choices)
    result = {"threshold": float(threshold), "alpha": float(alpha)}
    for year, d, y, route in ((2023, d23, y23, route23), (2024, d24, y24, route24)):
        mask = d["evaluate"]
        pred = d["dsf"] + (route >= threshold) * alpha * (d["factor"] - d["dsf"])
        result[str(year)] = {
            "dsf_bss": bss(y[mask], d["dsf"][mask]),
            "router_bss": bss(y[mask], pred[mask]),
            "delta_bss": bss(y[mask], pred[mask]) - bss(y[mask], d["dsf"][mask]),
            "routed_fraction": float(np.mean(route[mask] >= threshold)),
        }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
