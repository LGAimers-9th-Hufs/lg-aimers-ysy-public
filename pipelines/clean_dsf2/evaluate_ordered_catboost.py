#!/usr/bin/env python3
"""Evaluate genuinely ordered CatBoost candidates across 2023->2024."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from pipelines.clean_dsf2.evaluate_sparse_linear import choose
from pipelines.clean_forest.evaluate_residual_lookup import align, bss


TARGET = "control_success"
BASE_CATS = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]


def features(frame: pd.DataFrame, invariant: bool = False) -> tuple[pd.DataFrame, list[str]]:
    x = frame.drop(columns=[column for column in ("row_id", "season", TARGET) if column in frame]).copy()
    x["count_code"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["hand_match"] = (x["pitcher_hand"] == x["batter_hand"]).astype(str)
    x["inning_band"] = pd.cut(x["inning"], [-1, 3, 6, 9, 99], labels=False).astype(str)
    x["count_hand"] = x["count_code"] + "|" + x["pitcher_hand"].astype(str) + "|" + x["batter_hand"].astype(str)
    x["count_base"] = x["count_code"] + "|" + x["base_state"].astype(str)
    x["regime_count"] = x["game_type"].astype(str) + "|" + x["count_code"]
    x["p_logn"] = np.log1p(pd.to_numeric(x["asof_pitcher_n"], errors="coerce"))
    x["b_logn"] = np.log1p(pd.to_numeric(x["asof_batter_n"], errors="coerce"))
    p_n = pd.to_numeric(x["asof_pitcher_n"], errors="coerce").fillna(0.0)
    b_n = pd.to_numeric(x["asof_batter_n"], errors="coerce").fillna(0.0)
    x["p_sm200"] = (p_n * pd.to_numeric(x["asof_pitcher_success_rate"], errors="coerce").fillna(0.5) + 100.0) / (p_n + 200.0)
    x["b_sm200"] = (b_n * pd.to_numeric(x["asof_batter_success_rate"], errors="coerce").fillna(0.5) + 100.0) / (b_n + 200.0)
    x["p_form_shock"] = pd.to_numeric(x["asof_pitcher_prev1_game_success_rate"], errors="coerce") - pd.to_numeric(x["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    cats = [*BASE_CATS, "count_code", "hand_match", "inning_band", "count_hand", "count_base", "regime_count"]
    if invariant:
        x = x.drop(columns=["pitcher_id", "batter_id"])
        cats = [column for column in cats if column not in {"pitcher_id", "batter_id"}]
    for column in cats:
        x[column] = x[column].astype("string").fillna("<NA>").astype(str)
    return x, cats


def model(seed: int, depth: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE", boosting_type="Ordered", iterations=iterations,
        depth=depth, learning_rate=0.035, l2_leaf_reg=25.0,
        random_strength=0.5, bootstrap_type="Bayesian", bagging_temperature=0.5,
        random_seed=seed, thread_count=6, verbose=100,
        allow_writing_files=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--clean_oof_dir", default="/private/tmp/clean_moe_oof")
    parser.add_argument("--regime_oof", default="/private/tmp/regime_stack_2024_oof.npz")
    parser.add_argument("--output_dir", default="/private/tmp/clean_dsf2_catboost")
    parser.add_argument("--iterations", type=int, default=700)
    args = parser.parse_args()
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    train = frame.loc[frame["season"].eq(2023)].copy()
    valid = frame.loc[frame["season"].eq(2024)].copy()
    train_oof = align(Path(args.clean_oof_dir) / "clean_moe_2023.npz", train["row_id"].astype(str).to_numpy())
    valid_oof = align(Path(args.clean_oof_dir) / "clean_moe_2024.npz", valid["row_id"].astype(str).to_numpy())
    regime = align(Path(args.regime_oof), valid["row_id"].astype(str).to_numpy())
    base = np.clip(0.70 * valid_oof["prediction"] + 0.30 * regime["prediction"] - 0.002, 0.0, 1.0)
    y_train = train[TARGET].to_numpy(float)
    y = valid[TARGET].to_numpy(float)
    select = valid_oof["select"].astype(bool)
    evaluate = valid_oof["evaluate"].astype(bool)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    raw_predictions = {}
    for index, (name, invariant, depth) in enumerate((("full", False, 8), ("invariant", True, 7))):
        x_train, cats = features(train, invariant)
        x_valid, _ = features(valid, invariant)
        estimator = model(20260930 + index, depth, args.iterations)
        print(f"fit {name} rows={len(train):,} cols={x_train.shape[1]} cats={len(cats)}", flush=True)
        estimator.fit(x_train, y_train, cat_features=cats)
        raw = np.clip(estimator.predict(x_valid), 0.0, 1.0)
        raw_predictions[name] = raw
        params, candidate = choose(base, raw, y, select)
        candidates.append((np.mean((y[select] - candidate[select]) ** 2), name, params, candidate))
        estimator.save_model(output / f"{name}.cbm")
    _, name, params, candidate = min(candidates)
    report = {
        "selected": name, "blend": params,
        "select": {"base": bss(y, base, select), "candidate": bss(y, candidate, select)},
        "evaluate": {"base": bss(y, base, evaluate), "candidate": bss(y, candidate, evaluate)},
        "standalone": {key: bss(y, value, evaluate) for key, value in raw_predictions.items()},
        "segments": {},
        "residual_correlation": float(np.corrcoef(y[evaluate] - base[evaluate], y[evaluate] - raw_predictions[name][evaluate])[0, 1]),
        "rules": "Ordered CatBoost fit on 2023 official train; tune 2024 select; untouched 2024 evaluate; no external/test aggregation",
    }
    for game_type in ("F", "R"):
        mask = evaluate & valid["game_type"].astype(str).eq(game_type).to_numpy()
        report["segments"][game_type] = {"rows": int(mask.sum()), "base": bss(y, base, mask), "candidate": bss(y, candidate, mask)}
    np.savez_compressed(output / "predictions.npz", row_id=valid["row_id"].astype(str), y=y, base=base, candidate=candidate, select=select, evaluate=evaluate, **raw_predictions)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
