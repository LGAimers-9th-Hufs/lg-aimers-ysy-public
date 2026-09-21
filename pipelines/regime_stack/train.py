#!/usr/bin/env python3
"""Train a clean DSF-like regime stack from official train data only."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


TARGET = "control_success"
ID = "row_id"
ROOT = Path(__file__).resolve().parents[2]
REGIME_FEATURE_SOURCE = ROOT / "model_artifacts/factor_tabm_router/package/dsf/champion/regime_features.py"
ROUNDS = (30, 50, 70, 95, 130, 180, 230, 300, 380)


def load_feature_module():
    spec = importlib.util.spec_from_file_location("_clean_regime_features", REGIME_FEATURE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("regime feature loader is unavailable")
    spec.loader.exec_module(module)
    return module


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(prediction, float)) ** 2))


def bss(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, float)
    return 100000.0 * (1.0 - brier(y, prediction) / (y.mean() * (1.0 - y.mean())))


def category_maps(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    columns = [
        "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
        "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    ]
    result = {}
    for column in columns:
        values = sorted(frame[column].astype("string").dropna().unique().tolist())
        result[column] = {value: index for index, value in enumerate(values)}
    return result


def endpoint(frame: pd.DataFrame, entity: str, n_col: str, rate_cols: list[str]) -> dict[str, list[float]]:
    if not len(frame):
        return {}
    n = pd.to_numeric(frame[n_col], errors="coerce").fillna(0.0)
    index = n.groupby(frame[entity], sort=False).idxmax()
    selected = frame.loc[index, [entity, n_col, *rate_cols]]
    result = {}
    for row in selected.itertuples(index=False, name=None):
        count = float(0.0 if pd.isna(row[1]) else row[1])
        values = [count]
        values.extend(float(np.rint(count * (0.5 if pd.isna(value) else value))) for value in row[2:])
        result[str(row[0])] = values
    return result


def artifact(history: pd.DataFrame, maps: dict[str, dict[str, int]]) -> dict:
    return {
        "category_maps": maps,
        "pitcher_inseason_base": endpoint(
            history, "pitcher_id", "asof_pitcher_n", ["asof_pitcher_success_rate"]
        ),
        "batter_inseason_base": endpoint(
            history, "batter_id", "asof_batter_n",
            ["asof_batter_success_rate", "asof_batter_middle_rate"],
        ),
        "pitcher_kappa": 100.0,
        "batter_kappa": 100.0,
    }


def time_safe_history(frame: pd.DataFrame, maps: dict, feature_module, names: list[str]) -> pd.DataFrame:
    seasons = pd.to_numeric(frame["season"], errors="coerce").astype(int).to_numpy()
    blocks = []
    for season in sorted(np.unique(seasons)):
        mask = seasons == season
        frozen = artifact(frame.loc[seasons < season], maps)
        block = feature_module.build_features(frame.loc[mask], frozen, names)
        block.index = frame.index[mask]
        blocks.append(block)
        print(f"features season={season} rows={int(mask.sum()):,}", flush=True)
    return pd.concat(blocks).sort_index()


def feature_groups(feature_module) -> dict[str, list[str]]:
    full = list(feature_module.FULL_FEATURES)
    state = [
        name for name in full
        if name in feature_module.CONTEXT_FEATURES
        or name.startswith(("is_", "isb_", "success_", "middle_", "recent_", "shock_", "volatility_", "recovery_", "uncertainty_"))
    ]
    invariant = [name for name in full if name not in {"game_type_code", "pitcher_code", "batter_code"}]
    return {"specialist": full, "state": state, "invariant": invariant}


def fit_models(x: pd.DataFrame, y: np.ndarray, seasons: np.ndarray, groups: dict, feature_module) -> dict[str, lgb.LGBMRegressor]:
    latest = int(seasons.max())
    temporal_weights = np.power(0.50, latest - seasons).astype(np.float32)
    models = {}
    for index, (name, columns) in enumerate(groups.items()):
        rows = seasons == latest if name == "specialist" else np.ones(len(seasons), dtype=bool)
        weights = None if name == "specialist" else temporal_weights[rows]
        categorical = [column for column in columns if column in feature_module.CATEGORICAL_NAMES]
        model = lgb.LGBMRegressor(
            objective="regression_l2", n_estimators=max(ROUNDS), learning_rate=0.025,
            num_leaves=11, max_depth=-1, min_child_samples=1000,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
            reg_lambda=20.0, reg_alpha=1.0, max_bin=127, verbosity=-1,
            random_state=20260811 + index, n_jobs=6, force_col_wise=True,
        )
        print(f"fit {name} rows={int(rows.sum()):,} cols={len(columns)}", flush=True)
        model.fit(x.loc[rows, columns], y[rows], sample_weight=weights, categorical_feature=categorical)
        models[name] = model
    return models


def choose_rounds(models: dict, x: pd.DataFrame, y: np.ndarray, select: np.ndarray, groups: dict) -> tuple[dict, dict]:
    selected = {}
    predictions = {}
    for name, model in models.items():
        best = None
        for rounds in ROUNDS:
            prediction = model.predict(x[groups[name]], num_iteration=rounds)
            loss = brier(y[select], prediction[select])
            if best is None or loss < best[0]:
                best = (loss, rounds, prediction)
        selected[name] = int(best[1])
        predictions[name] = best[2]
    return selected, predictions


def choose_blend(predictions: dict, y: np.ndarray, select: np.ndarray) -> dict:
    matrix = np.column_stack([predictions[name] for name in ("specialist", "state", "invariant")])
    best = None
    for first in np.arange(0.0, 1.001, 0.05):
        for second in np.arange(0.0, 1.001 - first, 0.05):
            third = 1.0 - first - second
            raw = matrix @ np.array([first, second, third])
            for shift in np.arange(-0.025, 0.0051, 0.001):
                loss = brier(y[select], raw[select] + shift)
                if best is None or loss < best[0]:
                    best = (loss, first, second, third, shift)
    return {
        "specialist": float(best[1]), "state": float(best[2]),
        "invariant": float(best[3]), "shift": float(best[4]),
    }


def blend(predictions: dict, mixture: dict) -> np.ndarray:
    return np.clip(
        sum(mixture[name] * predictions[name] for name in ("specialist", "state", "invariant"))
        + mixture["shift"], 0.0, 1.0,
    )


def reference_masks(path: str, year: int, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path) / f"factor_{year}.npz", allow_pickle=False) as saved:
        order = pd.Series(np.arange(len(saved["row_id"])), index=saved["row_id"].astype(str)).reindex(row_ids.astype(str)).to_numpy()
        if not np.isfinite(order).all():
            raise RuntimeError("reference mask alignment failed")
        order = order.astype(int)
        return saved["select"][order].astype(bool), saved["evaluate"][order].astype(bool)


def evaluate(args, frame: pd.DataFrame, feature_module) -> dict:
    target_year = args.target_year
    history = frame.loc[frame["season"] < target_year].copy()
    validation = frame.loc[frame["season"] == target_year].copy()
    maps = category_maps(history)
    groups = feature_groups(feature_module)
    x_history = time_safe_history(history, maps, feature_module, groups["specialist"])
    frozen = artifact(history, maps)
    x_validation = feature_module.build_features(validation, frozen, groups["specialist"])
    models = fit_models(
        x_history, history[TARGET].to_numpy(np.float32),
        history["season"].to_numpy(int), groups, feature_module,
    )
    select, evaluate_mask = reference_masks(args.reference_oof, target_year, validation[ID].astype(str).to_numpy())
    rounds, predictions = choose_rounds(
        models, x_validation, validation[TARGET].to_numpy(float), select, groups
    )
    mixture = choose_blend(predictions, validation[TARGET].to_numpy(float), select)
    prediction = blend(predictions, mixture)
    y = validation[TARGET].to_numpy(float)
    report = {
        "target_year": target_year, "rounds": rounds, "mixture": mixture,
        "select_bss": bss(y[select], prediction[select]),
        "evaluate_bss": bss(y[evaluate_mask], prediction[evaluate_mask]),
        "components": {name: bss(y[evaluate_mask], pred[evaluate_mask]) for name, pred in predictions.items()},
        "segments": {},
        "rules": "official train only; prior-season training; select/evaluate disjoint; row-independent frozen artifacts",
    }
    for regime in ("F", "R"):
        mask = evaluate_mask & validation["game_type"].astype(str).eq(regime).to_numpy()
        report["segments"][regime] = {"rows": int(mask.sum()), "bss": bss(y[mask], prediction[mask])}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.prediction_output:
        np.savez_compressed(
            args.prediction_output,
            row_id=validation[ID].astype(str).to_numpy(), y=y, prediction=prediction,
            select=select, evaluate=evaluate_mask,
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--target_year", type=int, default=2024)
    parser.add_argument("--reference_oof", default="./factor_tabm_oof")
    parser.add_argument("--output", default="./regime_stack_results.json")
    parser.add_argument("--prediction_output")
    args = parser.parse_args()
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    evaluate(args, frame, load_feature_module())


if __name__ == "__main__":
    main()
