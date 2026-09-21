#!/usr/bin/env python3
"""Train and validate a row-local residual specialist for probe-free DSF OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OOF_DEFAULT = Path("<local>/DSF_CHAL061_FIX/oof/meta_blend_wf_2022_2024.npz")
DROP = {"row_id", "control_success", "season"}
CAT_MAPS = {
    "top_bottom": {"B": 0, "T": 1},
    "game_type": {"F": 0, "R": 1, "P": 0},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3, "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}


def features(frame: pd.DataFrame, medians: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    out: dict[str, np.ndarray] = {}
    for column in frame.columns:
        if column in DROP:
            continue
        if column in CAT_MAPS:
            out[column] = frame[column].astype(str).map(CAT_MAPS[column]).fillna(-1).to_numpy(np.float32)
        else:
            out[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(np.float32)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(np.float32)
    li = pd.to_numeric(frame["li"], errors="coerce").fillna(0).to_numpy(np.float32)
    runners = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy(np.float32)
    ps = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").fillna(0.5).to_numpy(np.float32)
    p1 = pd.to_numeric(frame["asof_pitcher_prev1_game_success_rate"], errors="coerce").fillna(pd.Series(ps)).to_numpy(np.float32)
    out["count_code_x"] = balls * 3.0 + strikes
    out["pressure_x"] = np.log1p(np.maximum(li, 0)) * (1.0 + runners) * (1.0 + (balls == 3))
    out["pitcher_form_x"] = p1 - ps
    out["pitcher_pressure_x"] = (ps - 0.5) * np.log1p(np.maximum(li, 0))
    x = pd.DataFrame(out, index=frame.index)
    if medians is None:
        medians = {c: float(x[c].median()) if np.isfinite(x[c].median()) else 0.0 for c in x.columns}
    x = x.fillna(pd.Series(medians)).fillna(0.0).astype(np.float32)
    return x, medians


def affine(p: np.ndarray, y: np.ndarray, index: np.ndarray) -> tuple[float, float]:
    a, b = np.linalg.lstsq(
        np.column_stack([np.ones(len(index)), p[index] - 0.5]), y[index] - 0.5, rcond=None
    )[0]
    return float(a), float(b)


def calibrated(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(0.5 + a + b * (p - 0.5), 0.0, 1.0)


def bss(y: np.ndarray, p: np.ndarray, index: np.ndarray) -> float:
    yy = y[index]
    return float(100000.0 * (1.0 - np.mean((yy - p[index]) ** 2) / (yy.mean() * (1.0 - yy.mean()))))


def params(seed: int) -> dict:
    return {
        "objective": "regression_l2", "metric": "l2", "learning_rate": 0.02,
        "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 2000,
        "lambda_l1": 0.5, "lambda_l2": 25.0, "feature_fraction": 0.75,
        "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
        "num_threads": 6, "seed": seed, "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--oof", type=Path, default=OOF_DEFAULT)
    parser.add_argument("--output", type=Path, default=ROOT / "model_artifacts/dsf_residual")
    args = parser.parse_args()
    z = np.load(args.oof, allow_pickle=False)
    mask = z["season"] == 2024
    ids = z["row_id"][mask].astype(str)
    prediction = z["blend_061"][mask].astype(np.float64)
    y = z["y"][mask].astype(np.float64)
    frame = pd.read_csv(args.train, encoding="utf-8-sig", low_memory=False)
    positions = pd.Series(np.arange(len(frame)), index=frame["row_id"].astype(str)).reindex(ids).to_numpy()
    if not np.isfinite(positions).all():
        raise RuntimeError("OOF row alignment failed")
    frame = frame.iloc[positions.astype(int)].reset_index(drop=True)
    x, medians = features(frame)
    blocks = np.array_split(np.arange(len(frame)), 6)
    rounds = (100, 200, 400)
    caps = (0.005, 0.010, 0.015, 0.020)
    strengths = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
    fold_records = []
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    for fold in range(1, 6):
        train_index = np.concatenate(blocks[:fold])
        valid_index = blocks[fold]
        a, b = affine(prediction, y, train_index)
        base = calibrated(prediction, a, b)
        residual = y[train_index] - base[train_index]
        booster = lgb.train(params(20261010 + fold), lgb.Dataset(x.iloc[train_index], label=residual), num_boost_round=max(rounds))
        for iteration in rounds:
            correction = booster.predict(x.iloc[valid_index], num_iteration=iteration)
            for cap in caps:
                clipped = np.clip(correction, -cap, cap)
                for strength in strengths:
                    candidate = base[valid_index] + strength * clipped
                    cache[(fold, hash((iteration, cap, strength)))] = (valid_index, candidate, bss(y, base, valid_index))
        fold_records.append({"fold": fold, "train_rows": len(train_index), "valid_rows": len(valid_index), "base_bss": bss(y, base, valid_index)})
        print(f"fold={fold} train={len(train_index):,} valid={len(valid_index):,}", flush=True)
    candidates = []
    for iteration in rounds:
        for cap in caps:
            for strength in strengths:
                deltas = []
                scores = []
                key = hash((iteration, cap, strength))
                for fold in range(1, 6):
                    idx, candidate, base_score = cache[(fold, key)]
                    yy = y[idx]
                    score = float(100000.0 * (1.0 - np.mean((yy - candidate) ** 2) / (yy.mean() * (1.0 - yy.mean()))))
                    deltas.append(score - base_score)
                    scores.append(score)
                candidates.append({"iterations": iteration, "cap": cap, "strength": strength, "deltas": deltas, "mean_delta": float(np.mean(deltas)), "min_delta": float(np.min(deltas)), "mean_score": float(np.mean(scores))})
    stable = [v for v in candidates if v["min_delta"] >= 0.0]
    selected = max(stable, key=lambda v: v["mean_delta"]) if stable else max(candidates, key=lambda v: v["mean_delta"])
    all_index = np.arange(len(frame))
    a, b = affine(prediction, y, all_index)
    base = calibrated(prediction, a, b)
    residual = y - base
    final = lgb.train(params(20261020), lgb.Dataset(x, label=residual), num_boost_round=selected["iterations"])
    args.output.mkdir(parents=True, exist_ok=True)
    final.save_model(str(args.output / "residual.txt"))
    artifact = {
        "feature_names": list(x.columns), "medians": medians,
        "affine": {"intercept": a, "slope": b}, "residual": selected,
        "validation": fold_records,
        "rules": "official 2024 train walk-forward DSF OOF only; chronological forward validation; row-local inference",
    }
    (args.output / "metadata.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "affine": artifact["affine"]}, indent=2))


if __name__ == "__main__":
    main()
