#!/usr/bin/env python3
"""Fit the frozen residual corrector from temporal OOF predictions only."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_factor_orthogonal_residual import prepare
from pipelines.dsf_upgrade.evaluate_trackman_safety_gate import fit_router, load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor_oof", default="./factor_tabm_oof_seed1")
    parser.add_argument("--track_oof", default="./factor_tabm_trackman_command_oof")
    parser.add_argument("--output_dir", default="./factor_tabm_final_seed1")
    args = parser.parse_args()
    factor = load(args.factor_oof, 2023)
    track = load(args.track_oof, 2023)
    if not np.array_equal(factor["row_id"], track["row_id"]):
        raise RuntimeError("OOF alignment mismatch for 2023")
    router = fit_router(factor)
    baseline, features = prepare(factor, track, router, 0.011956274509429932)
    residual_target = factor["y"].astype(float) - baseline
    model = lgb.LGBMRegressor(
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
    model.fit(features[factor["stop"]], residual_target[factor["stop"]])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "residual.joblib", compress=3)
    print(f"saved {output / 'residual.joblib'}")


if __name__ == "__main__":
    main()
