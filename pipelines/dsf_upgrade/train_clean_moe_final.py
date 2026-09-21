#!/usr/bin/env python3
"""Train the clean 2024 OOF residual MoE for 2025 deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_clean_2024_moe import aligned_frame, challenger_year
from pipelines.dsf_upgrade.evaluate_trackman_safety_gate import load
from pipelines.dsf_upgrade.residual_moe import make_moe_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--challenger_oof", required=True)
    parser.add_argument("--factor_dirs", nargs=3, default=["./factor_tabm_oof", "./factor_tabm_oof_seed1", "./factor_tabm_oof_seed2"])
    parser.add_argument("--track_oof", default="./factor_tabm_trackman_command_oof")
    parser.add_argument("--output_dir", default="./clean_moe_final")
    args = parser.parse_args()
    factor = [load(directory, 2024) for directory in args.factor_dirs]
    track = load(args.track_oof, 2024)
    reference = factor[0]
    for candidate in [*factor[1:], track]:
        if not np.array_equal(reference["row_id"], candidate["row_id"]):
            raise RuntimeError("2024 OOF alignment mismatch")
    challenger = challenger_year(args.challenger_oof, 2024, reference["row_id"])
    if not np.array_equal(reference["y"].astype(float), challenger["y"]):
        raise RuntimeError("2024 target alignment mismatch")
    frame = aligned_frame(args.train_path, 2024, reference["row_id"])
    base = challenger["prediction"]
    factor_matrix = np.column_stack([candidate["factor"] for candidate in factor]).astype(float)
    zeros = np.zeros(len(base), dtype=float)
    features = make_moe_features(
        frame, base, factor_matrix, track["factor"].astype(float), zeros, base, zeros
    )
    model = lgb.LGBMRegressor(
        objective="regression_l2", n_estimators=140, learning_rate=0.02, num_leaves=12,
        max_depth=4, min_child_samples=1400, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=60.0, verbosity=-1, random_state=99232, n_jobs=6,
    )
    model.fit(features, challenger["y"] - base)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "clean_moe.joblib", compress=3)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "alpha": 1.0,
                "cap": 0.04,
                "factor_seeds": [59025, 60025, 61025],
                "fit": "all 2024 strict walk-forward OOF after untouched evaluation",
                "selection": "2024 OOF stop fit, select hyperparameters, evaluate once",
                "provenance": "official train and 2019-2024 TrackMan; reproducible Challenger only",
                "excluded": "DSF champion leaderboard probes and untraceable anonymous-ID TrackMan mapping",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {output / 'clean_moe.joblib'} rows={len(base)} features={features.shape[1]}")


if __name__ == "__main__":
    main()
