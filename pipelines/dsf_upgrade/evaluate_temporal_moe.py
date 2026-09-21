#!/usr/bin/env python3
"""Cross-season clean residual evaluation: fit 2023 OOF, test 2024 OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_clean_2024_moe import aligned_frame, challenger_year
from pipelines.dsf_upgrade.evaluate_factor_router import brier, bss
from pipelines.dsf_upgrade.evaluate_trackman_safety_gate import load
from pipelines.dsf_upgrade.residual_moe import make_moe_features


CONFIGS = [
    {"name": "d1", "n_estimators": 100, "num_leaves": 2, "max_depth": 1, "min_child_samples": 3000, "reg_lambda": 100.0},
    {"name": "d2", "n_estimators": 120, "num_leaves": 5, "max_depth": 2, "min_child_samples": 3000, "reg_lambda": 100.0},
    {"name": "d3", "n_estimators": 140, "num_leaves": 7, "max_depth": 3, "min_child_samples": 2500, "reg_lambda": 100.0},
    {"name": "d4", "n_estimators": 140, "num_leaves": 12, "max_depth": 4, "min_child_samples": 2000, "reg_lambda": 120.0},
]


def load_year(args: argparse.Namespace, year: int) -> dict:
    factors = [load(directory, year) for directory in args.factor_dirs]
    track = load(args.track_oof, year)
    reference = factors[0]
    for candidate in [*factors[1:], track]:
        if not np.array_equal(reference["row_id"], candidate["row_id"]):
            raise RuntimeError(f"OOF alignment failed for {year}")
    challenger = challenger_year(args.challenger_oof, year, reference["row_id"])
    if not np.array_equal(reference["y"].astype(float), challenger["y"]):
        raise RuntimeError(f"target alignment failed for {year}")
    frame = aligned_frame(args.train_path, year, reference["row_id"])
    base = challenger["prediction"]
    factor_matrix = np.column_stack([candidate["factor"] for candidate in factors]).astype(float)
    zeros = np.zeros(len(base), dtype=float)
    features = make_moe_features(frame, base, factor_matrix, track["factor"], zeros, base, zeros)
    return {"frame": frame, "reference": reference, "y": challenger["y"], "base": base, "features": features}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--challenger_oof", required=True)
    parser.add_argument("--factor_dirs", nargs=3, default=["./factor_tabm_oof", "./factor_tabm_oof_seed1", "./factor_tabm_oof_seed2"])
    parser.add_argument("--track_oof", default="./factor_tabm_trackman_command_oof")
    parser.add_argument("--output", default="./temporal_moe_results.json")
    args = parser.parse_args()
    train = load_year(args, 2023)
    test = load_year(args, 2024)
    candidates = []
    predictions = {}
    target = train["y"] - train["base"]
    for index, config in enumerate(CONFIGS):
        params = {key: value for key, value in config.items() if key != "name"}
        model = lgb.LGBMRegressor(
            objective="regression_l2", learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
            verbosity=-1, random_state=77100 + index, n_jobs=6, **params,
        )
        model.fit(train["features"], target)
        raw = model.predict(test["features"])
        predictions[config["name"]] = raw
        for alpha in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
            for cap in (0.005, 0.010, 0.015, 0.020, 0.030, 0.040):
                prediction = test["base"] + alpha * np.clip(raw, -cap, cap)
                mask = test["reference"]["select"]
                candidates.append((brier(test["y"][mask], prediction[mask]), config["name"], alpha, cap))
    _, name, alpha, cap = min(candidates)
    prediction = test["base"] + alpha * np.clip(predictions[name], -cap, cap)
    evaluate = test["reference"]["evaluate"]
    report = {
        "fit_year": 2023,
        "test_year": 2024,
        "config": name,
        "alpha": alpha,
        "cap": cap,
        "base_bss": bss(test["y"][evaluate], test["base"][evaluate]),
        "candidate_bss": bss(test["y"][evaluate], prediction[evaluate]),
        "candidate_vs_base_bss": bss(test["y"][evaluate], prediction[evaluate]) - bss(test["y"][evaluate], test["base"][evaluate]),
        "rules": "fit 2023 OOF only; choose alpha/cap on 2024 select; report 2024 evaluate once; no DSF predictions",
    }
    report["segments"] = {}
    for regime in ("F", "R"):
        mask = evaluate & test["frame"]["game_type"].astype(str).eq(regime).to_numpy()
        report["segments"][regime] = {
            "rows": int(mask.sum()),
            "base_bss": bss(test["y"][mask], test["base"][mask]),
            "candidate_bss": bss(test["y"][mask], prediction[mask]),
        }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
