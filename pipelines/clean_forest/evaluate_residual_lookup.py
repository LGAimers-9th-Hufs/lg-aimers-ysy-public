#!/usr/bin/env python3
"""Chronological frozen train-residual lookups blended with clean_regime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


TARGET = "control_success"
GROUPS = {
    "pitcher": ["pitcher_id"],
    "batter": ["batter_id"],
    "pitcher_count": ["pitcher_id", "balls_before", "strikes_before"],
    "pitcher_regime": ["pitcher_id", "game_type"],
    "batter_count": ["batter_id", "balls_before", "strikes_before"],
    "pitcher_hand": ["pitcher_id", "batter_hand"],
    "pitcher_team_count": ["pitcher_team_id", "balls_before", "strikes_before"],
    "context": ["game_type", "balls_before", "strikes_before", "base_state", "pitcher_hand", "batter_hand"],
}
KAPPAS = (50.0, 200.0, 800.0)


def align(path: Path, row_ids: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        ids = saved["row_id"].astype(str)
        order = pd.Series(np.arange(len(ids)), index=ids).reindex(row_ids.astype(str)).to_numpy()
        if not np.isfinite(order).all():
            raise RuntimeError(f"alignment failed: {path}")
        order = order.astype(int)
        return {name: saved[name][order] for name in saved.files if len(saved[name]) == len(ids)}


def key_frame(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].astype("string").fillna("<NA>").agg("|".join, axis=1)


def lookup(train: pd.DataFrame, valid: pd.DataFrame, residual: np.ndarray, columns: list[str], kappa: float) -> np.ndarray:
    keys = key_frame(train, columns)
    stats = pd.DataFrame({"key": keys, "residual": residual}).groupby("key", sort=False)["residual"].agg(["sum", "count"])
    correction = stats["sum"] / (stats["count"] + kappa)
    return key_frame(valid, columns).map(correction).fillna(0.0).to_numpy(np.float32)


def bss(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    yy = y[mask]
    return float(100000.0 * (1.0 - np.mean((yy - p[mask]) ** 2) / (yy.mean() * (1.0 - yy.mean()))))


def split(ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bucket = np.array([int(hashlib.sha1(value.encode()).hexdigest()[:8], 16) % 5 for value in ids.astype(str)])
    return mask & (bucket != 0), mask & (bucket == 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--clean_oof_dir", default="/private/tmp/clean_moe_oof")
    parser.add_argument("--regime_oof", default="/private/tmp/regime_stack_2024_oof.npz")
    parser.add_argument("--output_dir", default="/private/tmp/clean_lookup_eval")
    args = parser.parse_args()
    columns = sorted({column for group in GROUPS.values() for column in group})
    usecols = ["row_id", "season", TARGET, *columns]
    frame = pd.read_csv(args.train_path, usecols=list(dict.fromkeys(usecols)), encoding="utf-8-sig", low_memory=False)
    train = frame.loc[frame["season"].eq(2023)].copy()
    valid = frame.loc[frame["season"].eq(2024)].copy()
    p23 = align(Path(args.clean_oof_dir) / "clean_moe_2023.npz", train["row_id"].astype(str).to_numpy())
    p24 = align(Path(args.clean_oof_dir) / "clean_moe_2024.npz", valid["row_id"].astype(str).to_numpy())
    regime = align(Path(args.regime_oof), valid["row_id"].astype(str).to_numpy())
    residual = train[TARGET].to_numpy(float) - p23["prediction"].astype(float)
    feature_names = []
    candidates = []
    for name, group in GROUPS.items():
        for kappa in KAPPAS:
            feature_names.append(f"{name}_k{int(kappa)}")
            candidates.append(lookup(train, valid, residual, group, kappa))
    x = np.column_stack(candidates).astype(np.float64)
    y = valid[TARGET].to_numpy(float)
    select = p24["select"].astype(bool)
    evaluate = p24["evaluate"].astype(bool)
    clean = np.clip(0.70 * p24["prediction"] + 0.30 * regime["prediction"] - 0.002, 0.0, 1.0)
    fit_mask, tune_mask = split(valid["row_id"].astype(str).to_numpy(), select)
    scale = x[fit_mask].std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = x / scale
    choices = []
    for alpha in (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
        trial = Ridge(alpha=alpha, fit_intercept=False)
        trial.fit(z[fit_mask], y[fit_mask] - clean[fit_mask])
        raw_trial = trial.predict(z)
        for strength in (0.25, 0.50, 0.75, 1.00):
            adjusted = np.clip(clean + strength * np.clip(raw_trial, -0.03, 0.03), 0.0, 1.0)
            loss = np.mean((y[tune_mask] - adjusted[tune_mask]) ** 2)
            choices.append((loss, alpha, strength))
    _, alpha, strength = min(choices)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(z[select], y[select] - clean[select])
    raw = model.predict(z)
    candidate = np.clip(clean + strength * np.clip(raw, -0.03, 0.03), 0.0, 1.0)
    report = {
        "alpha": alpha,
        "strength": strength,
        "feature_names": feature_names,
        "select": {"clean": bss(y, clean, select), "candidate": bss(y, candidate, select)},
        "evaluate": {"clean": bss(y, clean, evaluate), "candidate": bss(y, candidate, evaluate)},
        "segments": {},
        "correction_rms": float(np.sqrt(np.mean((candidate[evaluate] - clean[evaluate]) ** 2))),
        "rules": "2023 official train residual lookup; 2024 select-only ridge; untouched 2024 evaluate; row-independent inference",
    }
    for game_type in ("F", "R"):
        mask = evaluate & valid["game_type"].astype(str).eq(game_type).to_numpy()
        report["segments"][game_type] = {"rows": int(mask.sum()), "clean": bss(y, clean, mask), "candidate": bss(y, candidate, mask)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", row_id=valid["row_id"].astype(str), y=y, clean=clean, candidate=candidate, select=select, evaluate=evaluate)
    (output / "lookup_model.json").write_text(json.dumps({"feature_names": feature_names, "scale": scale.tolist(), "coef": model.coef_.tolist(), "alpha": alpha, "strength": strength, "groups": GROUPS, "kappas": KAPPAS}, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
