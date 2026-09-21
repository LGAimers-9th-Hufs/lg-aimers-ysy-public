#!/usr/bin/env python3
"""Strict clean MoE using only reproducible Challenger and walk-forward models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipelines.dsf_upgrade.evaluate_factor_router import brier, bss
from pipelines.dsf_upgrade.evaluate_trackman_safety_gate import load
from pipelines.dsf_upgrade.residual_moe import RAW_COLUMNS, make_moe_features


CONFIGS = [
    {"name": "d2", "n_estimators": 100, "num_leaves": 5, "max_depth": 2, "min_child_samples": 2500, "reg_lambda": 60.0},
    {"name": "d3", "n_estimators": 140, "num_leaves": 7, "max_depth": 3, "min_child_samples": 1800, "reg_lambda": 50.0},
    {"name": "d4", "n_estimators": 140, "num_leaves": 12, "max_depth": 4, "min_child_samples": 1400, "reg_lambda": 60.0},
    {"name": "d3_slow", "n_estimators": 220, "num_leaves": 7, "max_depth": 3, "min_child_samples": 2800, "reg_lambda": 100.0},
]


def challenger_year(path: str, year: int, row_ids: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        mask = saved["season"].astype(int) == year
        ids = saved["row_id"][mask].astype(str)
        order = pd.Series(np.arange(len(ids)), index=ids).reindex(row_ids.astype(str)).to_numpy()
        if not np.isfinite(order).all():
            raise RuntimeError(f"Challenger OOF alignment failed for {year}")
        order = order.astype(int)
        return {
            "y": saved["y"][mask][order].astype(float),
            "prediction": saved["challenger_selected"][mask][order].astype(float),
        }


def aligned_frame(train_path: str, year: int, row_ids: np.ndarray) -> pd.DataFrame:
    usecols = list(dict.fromkeys(["row_id", "season", "game_type", "top_bottom", "base_state", *RAW_COLUMNS]))
    frame = pd.read_csv(train_path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    frame = frame.loc[frame["season"].eq(year)].copy()
    frame["row_id"] = frame["row_id"].astype(str)
    frame = frame.set_index("row_id").reindex(row_ids.astype(str))
    if frame["season"].isna().any():
        raise RuntimeError(f"train alignment failed for {year}")
    return frame.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--challenger_oof", required=True)
    parser.add_argument("--factor_dirs", nargs=3, default=["./factor_tabm_oof", "./factor_tabm_oof_seed1", "./factor_tabm_oof_seed2"])
    parser.add_argument("--track_oof", default="./factor_tabm_trackman_command_oof")
    parser.add_argument("--output", default="./clean_moe_results.json")
    parser.add_argument("--prediction_output")
    parser.add_argument("--prediction_dir")
    args = parser.parse_args()
    report = {"folds": {}, "selection": None}
    selected = None

    for year in (2023, 2024):
        factor = [load(directory, year) for directory in args.factor_dirs]
        track = load(args.track_oof, year)
        reference = factor[0]
        for candidate in [*factor[1:], track]:
            if not np.array_equal(reference["row_id"], candidate["row_id"]):
                raise RuntimeError(f"factor alignment failed for {year}")
        challenger = challenger_year(args.challenger_oof, year, reference["row_id"])
        if not np.array_equal(reference["y"].astype(float), challenger["y"]):
            raise RuntimeError(f"target alignment failed for {year}")
        frame = aligned_frame(args.train_path, year, reference["row_id"])
        base = challenger["prediction"]
        factor_matrix = np.column_stack([candidate["factor"] for candidate in factor]).astype(float)
        zeros = np.zeros(len(base), dtype=float)
        features = make_moe_features(
            frame, base, factor_matrix, track["factor"].astype(float), zeros, base, zeros
        )
        y = challenger["y"]
        target = y - base
        choices = []
        for index, config in enumerate(CONFIGS):
            params = {key: value for key, value in config.items() if key != "name"}
            model = lgb.LGBMRegressor(
                objective="regression_l2", learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
                verbosity=-1, random_state=99230 + index, n_jobs=6, **params,
            )
            model.fit(features[reference["stop"]], target[reference["stop"]])
            raw = model.predict(features)
            for alpha in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
                for cap in (0.005, 0.010, 0.015, 0.020, 0.030, 0.040):
                    prediction = base + alpha * np.clip(raw, -cap, cap)
                    mask = reference["select"]
                    choices.append((brier(y[mask], prediction[mask]), config["name"], alpha, cap, raw))
        if year == 2024:
            _, name, alpha, cap, raw = min(choices, key=lambda item: item[0])
            selected = {"config": name, "alpha": alpha, "cap": cap}
        else:
            # Reverse validation uses its own select rows only as a stability diagnostic.
            _, name, alpha, cap, raw = min(choices, key=lambda item: item[0])
        prediction = base + alpha * np.clip(raw, -cap, cap)
        if args.prediction_dir:
            prediction_dir = Path(args.prediction_dir)
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_dir / f"clean_moe_{year}.npz",
                row_id=reference["row_id"].astype(str), y=y, prediction=prediction,
                stop=reference["stop"], select=reference["select"], evaluate=reference["evaluate"],
            )
        evaluate = reference["evaluate"]
        base_score = bss(y[evaluate], base[evaluate])
        candidate_score = bss(y[evaluate], prediction[evaluate])
        fold = {
            "config": name,
            "alpha": alpha,
            "cap": cap,
            "base_bss": base_score,
            "candidate_bss": candidate_score,
            "candidate_vs_base_bss": candidate_score - base_score,
        }
        if year == 2024:
            segment_values = {
                "game_type_F": frame["game_type"].astype(str).eq("F").to_numpy(),
                "game_type_R": frame["game_type"].astype(str).eq("R").to_numpy(),
                "pitcher_n_low": pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).lt(200).to_numpy(),
                "pitcher_n_mid": pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).between(200, 1499).to_numpy(),
                "pitcher_n_high": pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).ge(1500).to_numpy(),
            }
            fold["segments"] = {}
            for segment, values in segment_values.items():
                mask = evaluate & values
                fold["segments"][segment] = {
                    "rows": int(mask.sum()),
                    "candidate_vs_base_bss": bss(y[mask], prediction[mask]) - bss(y[mask], base[mask]),
                }
            if args.prediction_output:
                np.savez_compressed(
                    args.prediction_output,
                    row_id=reference["row_id"].astype(str), y=y, prediction=prediction,
                    select=reference["select"], evaluate=reference["evaluate"],
                )
        report["folds"][str(year)] = fold
    report["selection"] = selected
    report["rules"] = (
        "official train and TrackMan only; reproducible Challenger OOF; strict past-season FactorTabM OOF; "
        "no DSF leaderboard-probe artifacts"
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
