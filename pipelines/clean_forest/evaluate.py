#!/usr/bin/env python3
"""Evaluate clean temporal ExtraTrees on a frozen 2024 select/evaluate split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from pipelines.regime_stack.train import (
    ID, TARGET, artifact, category_maps, load_feature_module, time_safe_history,
)


FEATURE_DROP = {"pitcher_code", "batter_code"}


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def bss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    return 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean())))


def align_npz(path: Path, row_ids: np.ndarray) -> dict[str, np.ndarray]:
    # These OOF files are generated locally by the tracked training pipelines. Some
    # older runs stored row_id as an object array, which requires pickle to decode.
    with np.load(path, allow_pickle=True) as saved:
        source_ids = saved["row_id"].astype(str)
        order = pd.Series(np.arange(len(source_ids)), index=source_ids).reindex(row_ids.astype(str)).to_numpy()
        if not np.isfinite(order).all():
            raise RuntimeError(f"row alignment failed for {path}")
        order = order.astype(int)
        return {name: saved[name][order] for name in saved.files if len(saved[name]) == len(source_ids)}


def matrix(frame: pd.DataFrame, medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    if medians is None:
        medians = np.nanmedian(values, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
    missing = np.where(np.isnan(values))
    values[missing] = medians[missing[1]]
    return values, medians


def fit_candidate(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    latest_only: bool,
    leaf: int,
    depth: int | None,
    trees: int,
    seed: int,
) -> ExtraTreesRegressor:
    latest = int(seasons.max())
    rows = seasons == latest if latest_only else seasons >= latest - 2
    weights = None if latest_only else np.power(0.55, latest - seasons[rows]).astype(np.float32)
    model = ExtraTreesRegressor(
        n_estimators=trees,
        max_depth=depth,
        min_samples_leaf=leaf,
        max_features=0.78,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )
    print(f"fit {name} rows={int(rows.sum()):,} trees={trees} leaf={leaf} depth={depth}", flush=True)
    model.fit(x[rows], y[rows], sample_weight=weights)
    return model


def choose_forest_blend(predictions: dict[str, np.ndarray], y: np.ndarray, select: np.ndarray) -> dict:
    names = list(predictions)
    best = None
    # Convex two-member blend. This deliberately avoids free test/leaderboard calibration.
    for weight in np.arange(0.0, 1.001, 0.05):
        p = weight * predictions[names[0]] + (1.0 - weight) * predictions[names[1]]
        loss = brier(y[select], p[select])
        if best is None or loss < best[0]:
            best = (loss, weight)
    return {names[0]: float(best[1]), names[1]: float(1.0 - best[1])}


def choose_final(clean: np.ndarray, forest: np.ndarray, y: np.ndarray, select: np.ndarray) -> dict:
    best = None
    for weight in np.arange(0.0, 0.801, 0.025):
        for shift in np.arange(-0.010, 0.0041, 0.001):
            p = np.clip((1.0 - weight) * clean + weight * forest + shift, 0.0, 1.0)
            loss = brier(y[select], p[select])
            if best is None or loss < best[0]:
                best = (loss, weight, shift)
    return {"forest_weight": float(best[1]), "shift": float(best[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--clean_oof", default="/private/tmp/clean_moe_2024_oof.npz")
    parser.add_argument("--regime_oof", default="/private/tmp/regime_stack_2024_oof.npz")
    parser.add_argument("--output_dir", default="/private/tmp/clean_forest_eval")
    parser.add_argument("--trees", type=int, default=240)
    parser.add_argument("--reuse_models", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    history = frame.loc[frame["season"] < 2024].copy()
    valid = frame.loc[frame["season"] == 2024].copy()
    feature_module = load_feature_module()
    features = [name for name in feature_module.FULL_FEATURES if name not in FEATURE_DROP]
    maps = category_maps(history)
    x_history_frame = time_safe_history(history, maps, feature_module, features)
    x_valid_frame = feature_module.build_features(valid, artifact(history, maps), features)
    x_history, medians = matrix(x_history_frame)
    x_valid, _ = matrix(x_valid_frame, medians)
    y_history = history[TARGET].to_numpy(np.float32)
    seasons = history["season"].to_numpy(int)
    y = valid[TARGET].to_numpy(np.float32)

    configs = {
        "recent": dict(latest_only=True, leaf=80, depth=26, seed=20260831),
        "temporal": dict(latest_only=False, leaf=160, depth=28, seed=20260832),
    }
    models = {}
    predictions = {}
    for name, config in configs.items():
        model_path = output / f"{name}.joblib"
        if args.reuse_models and model_path.is_file():
            print(f"load {name} from {model_path}", flush=True)
            model = joblib.load(model_path)
        else:
            model = fit_candidate(name, x_history, y_history, seasons, trees=args.trees, **config)
        models[name] = model
        predictions[name] = np.clip(model.predict(x_valid), 0.0, 1.0)
        if not model_path.is_file():
            joblib.dump(model, model_path, compress=3)

    aligned_clean = align_npz(Path(args.clean_oof), valid[ID].astype(str).to_numpy())
    aligned_regime = align_npz(Path(args.regime_oof), valid[ID].astype(str).to_numpy())
    select = aligned_clean["select"].astype(bool)
    evaluate = aligned_clean["evaluate"].astype(bool)
    if not np.array_equal(select, aligned_regime["select"].astype(bool)) or not np.array_equal(
        evaluate, aligned_regime["evaluate"].astype(bool)
    ):
        raise RuntimeError("clean/regime masks differ")
    clean = np.clip(0.70 * aligned_clean["prediction"] + 0.30 * aligned_regime["prediction"] - 0.002, 0.0, 1.0)
    forest_mix = choose_forest_blend(predictions, y, select)
    forest = sum(forest_mix[name] * predictions[name] for name in predictions)
    final = choose_final(clean, forest, y, select)
    candidate = np.clip(
        (1.0 - final["forest_weight"]) * clean + final["forest_weight"] * forest + final["shift"],
        0.0, 1.0,
    )
    report = {
        "features": features,
        "configs": configs,
        "forest_mix": forest_mix,
        "final": final,
        "select": {"clean": bss(y[select], clean[select]), "forest": bss(y[select], forest[select]), "candidate": bss(y[select], candidate[select])},
        "evaluate": {"clean": bss(y[evaluate], clean[evaluate]), "forest": bss(y[evaluate], forest[evaluate]), "candidate": bss(y[evaluate], candidate[evaluate])},
        "segments": {},
        "residual_correlation": float(np.corrcoef(y[evaluate] - clean[evaluate], y[evaluate] - forest[evaluate])[0, 1]),
        "rules": "official train only; frozen prior-season features; no test aggregation, external data, or leaderboard fitting",
    }
    for game_type in ("F", "R"):
        mask = evaluate & valid["game_type"].astype(str).eq(game_type).to_numpy()
        report["segments"][game_type] = {
            "rows": int(mask.sum()),
            "clean": bss(y[mask], clean[mask]),
            "forest": bss(y[mask], forest[mask]),
            "candidate": bss(y[mask], candidate[mask]),
        }
    np.savez_compressed(
        output / "predictions.npz", row_id=valid[ID].astype(str).to_numpy(), y=y,
        clean=clean, forest=forest, candidate=candidate, select=select, evaluate=evaluate,
        recent=predictions["recent"], temporal=predictions["temporal"], medians=medians,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
