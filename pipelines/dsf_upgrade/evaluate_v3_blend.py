#!/usr/bin/env python3
"""DSF/Challenger와 V3의 strict recent walk-forward OOF 결합 평가."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipelines.lightgbm_v3.train import TARGET_COL, params, temporal_features


YEARS = (2022, 2023, 2024)
VARIANTS = 3


def raw_bss(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.001, 0.999)
    brier = float(np.mean((p - y) ** 2))
    ref = float(y.mean() * (1.0 - y.mean()))
    return 100000.0 * (1.0 - brier / ref), brier


def convex_weights(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[1]
    result = minimize(
        lambda w: float(np.mean((matrix @ w - y) ** 2)),
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    return weights / weights.sum()


def fit_v3_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    y: np.ndarray,
    rounds: list[int],
) -> dict[int, np.ndarray]:
    season = frame["season"].to_numpy(int)
    output: dict[int, np.ndarray] = {}
    for year in YEARS:
        train_mask = season < year
        valid_mask = season == year
        sample_year = season[train_mask]
        sample_weight = (1.0 + 0.18 * (sample_year - sample_year.min())).astype(np.float32)
        predictions = []
        for variant in range(VARIANTS):
            model = lgb.LGBMClassifier(**params(142 + variant * 97, variant, rounds[variant]))
            model.fit(features.loc[train_mask], y[train_mask], sample_weight=sample_weight)
            predictions.append(model.predict_proba(features.loc[valid_mask])[:, 1])
        output[year] = np.mean(np.column_stack(predictions), axis=1)
        print(f"V3 fold={year} rows={valid_mask.sum():,} score={raw_bss(y[valid_mask], output[year])[0]:.3f}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--dsf_oof", required=True)
    parser.add_argument("--v3_artifact", default="./model_v3/artifacts.json")
    parser.add_argument("--cache", default="./dsf_upgrade_v3_oof.npz")
    parser.add_argument("--out", default="./dsf_upgrade_v3_report.json")
    parser.add_argument("--reuse_v3", action="store_true")
    args = parser.parse_args()
    started = time.time()

    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    frame["row_id"] = frame["row_id"].astype(str)
    y = frame[TARGET_COL].to_numpy(np.float64)
    cache = Path(args.cache)
    if args.reuse_v3 and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            v3 = {year: saved[f"pred_{year}"].astype(np.float64) for year in YEARS}
            cached_ids = {year: saved[f"row_id_{year}"].astype(str) for year in YEARS}
        for year in YEARS:
            expected = frame.loc[frame["season"] == year, "row_id"].to_numpy()
            if not np.array_equal(expected, cached_ids[year]):
                raise RuntimeError(f"V3 cache row_id mismatch: {year}")
    else:
        artifact = json.loads(Path(args.v3_artifact).read_text(encoding="utf-8"))
        rounds = [int(v) for v in artifact["validation"]["best_rounds"]]
        features = temporal_features(frame)
        v3 = fit_v3_oof(frame, features, y, rounds)
        np.savez_compressed(
            cache,
            **{f"row_id_{year}": frame.loc[frame["season"] == year, "row_id"].to_numpy(dtype=str) for year in YEARS},
            **{f"pred_{year}": v3[year].astype(np.float32) for year in YEARS},
        )

    with np.load(args.dsf_oof, allow_pickle=False) as saved:
        dsf = pd.DataFrame({
            "row_id": saved["row_id"].astype(str),
            "origin": saved["origin"].astype(str),
            "y_saved": saved["y"].astype(np.float64),
            "champion": saved["champion_reconstructed"].astype(np.float64),
            "challenger": saved["challenger"].astype(np.float64),
            "blend_061": saved["blend_061"].astype(np.float64),
        })

    parts = []
    for year in YEARS:
        local = frame.loc[frame["season"] == year, ["row_id", TARGET_COL]].copy()
        local["v3"] = v3[year]
        local = local.merge(dsf, on="row_id", how="inner", validate="one_to_one")
        # The recovered DSF reference contains only the released late-2022
        # holdout, while 2023/2024 cover their complete seasons.
        if year in (2023, 2024) and len(local) != int((frame["season"] == year).sum()):
            raise RuntimeError(f"DSF OOF coverage mismatch: {year} rows={len(local)}")
        if not np.array_equal(local[TARGET_COL].to_numpy(np.float64), local["y_saved"].to_numpy(np.float64)):
            raise RuntimeError(f"target mismatch: {year}")
        local["season"] = year
        parts.append(local)
    aligned = pd.concat(parts, ignore_index=True)

    names = ["champion", "challenger", "v3"]
    matrix = aligned[names].to_numpy(np.float64)
    target = aligned[TARGET_COL].to_numpy(np.float64)
    season = aligned["season"].to_numpy(int)
    residual_corr = {}
    individual = {}
    for year in YEARS:
        mask = season == year
        residual_corr[str(year)] = np.corrcoef(matrix[mask] - target[mask, None], rowvar=False).tolist()
        individual[str(year)] = {name: raw_bss(target[mask], matrix[mask, i]) for i, name in enumerate(names)}
        individual[str(year)]["blend_061"] = raw_bss(target[mask], aligned.loc[mask, "blend_061"])

    fit22 = season == 2022
    fit23 = season == 2023
    eval24 = season == 2024
    weights_23 = convex_weights(target[fit23], matrix[fit23])
    pred24 = matrix[eval24] @ weights_23

    # Robust grid: DSF champion remains >=80%; choose by 2023 Brier and report 2024 untouched.
    grid = []
    for challenger_weight in np.arange(0.0, 0.201, 0.01):
        for v3_weight in np.arange(0.0, 0.201, 0.01):
            champion_weight = 1.0 - challenger_weight - v3_weight
            if champion_weight < 0.80:
                continue
            weights = np.array([champion_weight, challenger_weight, v3_weight])
            p23 = matrix[fit23] @ weights
            p24 = matrix[eval24] @ weights
            grid.append((raw_bss(target[fit23], p23)[1], weights, raw_bss(target[eval24], p24)[0]))
    grid.sort(key=lambda item: item[0])
    robust_weights = grid[0][1]

    # Pre-2024 stability choice: average normalized Brier across the independent
    # 2022 holdout and full 2023 origin, then evaluate once on untouched 2024.
    stable_grid = []
    for challenger_weight in np.arange(0.0, 0.201, 0.01):
        for v3_weight in np.arange(0.0, 0.301, 0.01):
            champion_weight = 1.0 - challenger_weight - v3_weight
            if champion_weight < 0.70:
                continue
            weights = np.array([champion_weight, challenger_weight, v3_weight])
            losses = []
            for mask in (fit22, fit23):
                score, _ = raw_bss(target[mask], matrix[mask] @ weights)
                losses.append(1.0 - score / 100000.0)
            stable_grid.append((0.4 * losses[0] + 0.6 * losses[1], weights))
    stable_grid.sort(key=lambda item: item[0])
    stable_weights = stable_grid[0][1]

    report = {
        "pipeline": "DSF champion + Challenger + V3 strict OOF",
        "rules": "official train OOF only; fixed row-aligned probabilities; no test/leaderboard-derived fitting",
        "caveat": "champion_reconstructed was algebraically recovered from a frozen historical blend",
        "rows": {str(year): int((season == year).sum()) for year in YEARS},
        "individual": individual,
        "residual_correlation": residual_corr,
        "fit_2023_convex_weights": dict(zip(names, weights_23.tolist())),
        "fit_2023_eval_2024": raw_bss(target[eval24], pred24),
        "robust_grid_weights": dict(zip(names, robust_weights.tolist())),
        "robust_grid_fit_2023": raw_bss(target[fit23], matrix[fit23] @ robust_weights),
        "robust_grid_eval_2024": raw_bss(target[eval24], matrix[eval24] @ robust_weights),
        "stable_pre2024_weights": dict(zip(names, stable_weights.tolist())),
        "stable_pre2024_scores": {
            "2022": raw_bss(target[fit22], matrix[fit22] @ stable_weights),
            "2023": raw_bss(target[fit23], matrix[fit23] @ stable_weights),
            "2024_untouched": raw_bss(target[eval24], matrix[eval24] @ stable_weights),
        },
        "elapsed_seconds": time.time() - started,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
