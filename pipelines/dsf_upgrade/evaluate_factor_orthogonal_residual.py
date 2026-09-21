#!/usr/bin/env python3
"""Test a small residual corrector after the FactorTabM/TrackMan safety gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_factor_router import brier, bss, router_features
from pipelines.dsf_upgrade.evaluate_trackman_safety_gate import fit_router, load


def prepare(
    factor: dict[str, np.ndarray],
    track: dict[str, np.ndarray],
    router: lgb.LGBMClassifier,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    dsf = factor["dsf"].astype(float)
    factor_prediction = factor["factor"].astype(float)
    track_prediction = track["factor"].astype(float)
    route_probability = router.predict_proba(router_features(factor))[:, 1]
    correction = 0.30 * (factor_prediction - dsf)
    routed = route_probability >= 0.50
    track_delta = track_prediction - dsf
    unsafe = routed & (correction * track_delta < 0.0) & (np.abs(track_delta) >= cutoff)
    baseline = dsf + routed * correction
    baseline[unsafe] = dsf[unsafe]
    features = np.column_stack(
        [
            dsf,
            factor_prediction,
            track_prediction,
            factor_prediction - dsf,
            track_delta,
            np.abs(factor_prediction - dsf),
            np.abs(track_delta),
            route_probability,
            routed.astype(float),
            unsafe.astype(float),
            (factor_prediction - dsf) * track_delta,
            np.abs(dsf - 0.5),
        ]
    )
    return baseline, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor_oof", default="./factor_tabm_oof_seed1")
    parser.add_argument("--track_oof", default="./factor_tabm_trackman_command_oof")
    parser.add_argument("--output", default="./orthogonal_residual_results.json")
    args = parser.parse_args()
    factor = {year: load(args.factor_oof, year) for year in (2023, 2024)}
    track = {year: load(args.track_oof, year) for year in (2023, 2024)}
    for year in (2023, 2024):
        if not np.array_equal(factor[year]["row_id"], track[year]["row_id"]):
            raise RuntimeError(f"OOF alignment mismatch for {year}")

    cutoff = 0.011956274509429932
    router = fit_router(factor[2023])
    prepared = {year: prepare(factor[year], track[year], router, cutoff) for year in (2023, 2024)}
    baseline23, features23 = prepared[2023]
    residual_target = factor[2023]["y"].astype(float) - baseline23
    residual = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=80,
        learning_rate=0.02,
        num_leaves=5,
        max_depth=2,
        min_child_samples=2500,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=50.0,
        verbosity=-1,
        random_state=77203,
        n_jobs=6,
    )
    residual.fit(features23[factor[2023]["stop"]], residual_target[factor[2023]["stop"]])
    residual_predictions = {year: residual.predict(prepared[year][1]) for year in (2023, 2024)}

    candidates = []
    for alpha in (0.0, 0.05, 0.10, 0.20, 0.30):
        for cap in (0.0025, 0.0050, 0.0100):
            season_losses = []
            for year in (2023, 2024):
                d = factor[year]
                mask = d["select"]
                prediction = prepared[year][0] + alpha * np.clip(residual_predictions[year], -cap, cap)
                reference = d["y"][mask].mean() * (1.0 - d["y"][mask].mean())
                season_losses.append(brier(d["y"][mask], prediction[mask]) / reference)
            candidates.append((float(np.mean(season_losses)), alpha, cap))
    _, alpha, cap = min(candidates)

    result = {"residual_alpha": alpha, "residual_cap": cap, "years": {}}
    for year in (2023, 2024):
        d = factor[year]
        mask = d["evaluate"]
        baseline = prepared[year][0]
        prediction = baseline + alpha * np.clip(residual_predictions[year], -cap, cap)
        base_score = bss(d["y"][mask], baseline[mask])
        residual_score = bss(d["y"][mask], prediction[mask])
        result["years"][str(year)] = {
            "safety_gate_delta_bss": base_score - bss(d["y"][mask], d["dsf"][mask]),
            "residual_delta_bss": residual_score - bss(d["y"][mask], d["dsf"][mask]),
            "residual_vs_safety_bss": residual_score - base_score,
        }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
