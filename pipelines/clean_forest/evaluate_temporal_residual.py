#!/usr/bin/env python3
"""Fit clean residual structure in 2023 and apply it forward to clean_regime in 2024."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_clean_2024_moe import aligned_frame
from pipelines.dsf_upgrade.evaluate_trackman_safety_gate import load
from pipelines.dsf_upgrade.residual_moe import make_moe_features
from pipelines.clean_forest.evaluate_residual_lookup import align, bss


CONFIGS = [
    {"name": "d1", "n_estimators": 100, "num_leaves": 2, "max_depth": 1, "min_child_samples": 4000, "reg_lambda": 150.0},
    {"name": "d2", "n_estimators": 140, "num_leaves": 5, "max_depth": 2, "min_child_samples": 3500, "reg_lambda": 150.0},
    {"name": "d3", "n_estimators": 160, "num_leaves": 7, "max_depth": 3, "min_child_samples": 3000, "reg_lambda": 180.0},
]


def year_data(args: argparse.Namespace, year: int) -> dict:
    factors = [load(directory, year) for directory in args.factor_dirs]
    track = load(args.track_oof, year)
    reference = factors[0]
    clean = align(Path(args.clean_oof_dir) / f"clean_moe_{year}.npz", reference["row_id"])
    frame = aligned_frame(args.train_path, year, reference["row_id"])
    factor_matrix = np.column_stack([candidate["factor"] for candidate in factors]).astype(float)
    zeros = np.zeros(len(frame), dtype=float)
    features = make_moe_features(
        frame, clean["prediction"], factor_matrix, track["factor"], zeros,
        clean["prediction"], zeros,
    )
    return {"reference": reference, "clean": clean, "frame": frame, "features": features}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--clean_oof_dir", default="/private/tmp/clean_moe_oof")
    parser.add_argument("--regime_oof", default="/private/tmp/regime_stack_2024_oof.npz")
    parser.add_argument("--factor_dirs", nargs=3, default=["./factor_tabm_oof", "./factor_tabm_oof_seed1", "./factor_tabm_oof_seed2"])
    parser.add_argument("--track_oof", default="./factor_tabm_trackman_command_oof")
    parser.add_argument("--output_dir", default="/private/tmp/clean_temporal_eval")
    args = parser.parse_args()
    train = year_data(args, 2023)
    valid = year_data(args, 2024)
    regime = align(Path(args.regime_oof), valid["reference"]["row_id"])
    y_train = train["clean"]["y"].astype(float)
    y = valid["clean"]["y"].astype(float)
    base = np.clip(0.70 * valid["clean"]["prediction"] + 0.30 * regime["prediction"] - 0.002, 0.0, 1.0)
    target = y_train - train["clean"]["prediction"]
    select = valid["clean"]["select"].astype(bool)
    evaluate = valid["clean"]["evaluate"].astype(bool)
    choices = []
    raw_predictions = {}
    models = {}
    for index, config in enumerate(CONFIGS):
        params = {key: value for key, value in config.items() if key != "name"}
        model = lgb.LGBMRegressor(
            objective="regression_l2", learning_rate=0.02, subsample=0.8,
            colsample_bytree=0.8, verbosity=-1, random_state=20260910 + index,
            n_jobs=6, **params,
        )
        model.fit(train["features"], target)
        raw = model.predict(valid["features"])
        models[config["name"]] = model
        raw_predictions[config["name"]] = raw
        for strength in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
            for cap in (0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030):
                for shift in (-0.002, -0.001, 0.0, 0.001, 0.002):
                    candidate = np.clip(base + strength * np.clip(raw, -cap, cap) + shift, 0.0, 1.0)
                    loss = np.mean((y[select] - candidate[select]) ** 2)
                    choices.append((loss, config["name"], strength, cap, shift))
    _, name, strength, cap, shift = min(choices)
    raw = raw_predictions[name]
    candidate = np.clip(base + strength * np.clip(raw, -cap, cap) + shift, 0.0, 1.0)
    report = {
        "config": name, "strength": strength, "cap": cap, "shift": shift,
        "select": {"clean": bss(y, base, select), "candidate": bss(y, candidate, select)},
        "evaluate": {"clean": bss(y, base, evaluate), "candidate": bss(y, candidate, evaluate)},
        "segments": {},
        "correction_rms": float(np.sqrt(np.mean((candidate[evaluate] - base[evaluate]) ** 2))),
        "rules": "fit 2023 official OOF residual; tune on 2024 select; untouched 2024 evaluate; no leaderboard or test aggregation",
    }
    for game_type in ("F", "R"):
        mask = evaluate & valid["frame"]["game_type"].astype(str).eq(game_type).to_numpy()
        report["segments"][game_type] = {"rows": int(mask.sum()), "clean": bss(y, base, mask), "candidate": bss(y, candidate, mask)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models[name].booster_.save_model(str(output / "temporal_residual.txt"))
    np.savez_compressed(output / "predictions.npz", row_id=valid["reference"]["row_id"], y=y, clean=base, candidate=candidate, raw=raw, select=select, evaluate=evaluate)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
