#!/usr/bin/env python3
"""Crossed pitcher/batter/context effects with exact sparse ridge fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipelines.clean_dsf2.evaluate_sparse_linear import choose
from pipelines.clean_forest.evaluate_residual_lookup import align, bss


TARGET = "control_success"
ALPHAS = (30.0, 100.0, 300.0, 1000.0, 3000.0)
NUMERIC = [
    "li", "home_win_expectancy", "score_diff_pitcher_team",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def categorical(frame: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    count = frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    hand = frame["pitcher_hand"].astype(str) + "-" + frame["batter_hand"].astype(str)
    pitcher = frame["pitcher_id"].astype(str)
    batter = frame["batter_id"].astype(str)
    x["count"] = count
    x["context"] = frame["game_type"].astype(str) + "|" + count + "|" + frame["base_state"].astype(str) + "|" + hand
    x["inning_context"] = frame["inning"].astype(str) + "|" + frame["num_runners_on"].astype(str) + "|" + frame["outs_before"].astype(str)
    x["pitcher"] = pitcher
    x["batter"] = batter
    x["pitcher_count"] = pitcher + "|" + count
    x["batter_count"] = batter + "|" + count
    x["pitcher_bhand"] = pitcher + "|" + frame["batter_hand"].astype(str)
    x["pitcher_regime"] = pitcher + "|" + frame["game_type"].astype(str)
    x["pteam_count"] = frame["pitcher_team_id"].astype(str) + "|" + count
    x["bteam_count"] = frame["batter_team_id"].astype(str) + "|" + count
    x["month_dow"] = frame["game_month"].astype(str) + "|" + frame["game_dayofweek"].astype(str)
    return x


def numeric(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    a = train[NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    b = valid[NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    a[:, NUMERIC.index("asof_pitcher_n")] = np.log1p(np.maximum(a[:, NUMERIC.index("asof_pitcher_n")], 0.0))
    b[:, NUMERIC.index("asof_pitcher_n")] = np.log1p(np.maximum(b[:, NUMERIC.index("asof_pitcher_n")], 0.0))
    a[:, NUMERIC.index("asof_batter_n")] = np.log1p(np.maximum(a[:, NUMERIC.index("asof_batter_n")], 0.0))
    b[:, NUMERIC.index("asof_batter_n")] = np.log1p(np.maximum(b[:, NUMERIC.index("asof_batter_n")], 0.0))
    medians = np.nanmedian(a, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    for values in (a, b):
        missing = np.where(~np.isfinite(values))
        values[missing] = medians[missing[1]]
    scaler = StandardScaler()
    return sparse.csr_matrix(scaler.fit_transform(a)), sparse.csr_matrix(scaler.transform(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--clean_oof_dir", default="/private/tmp/clean_moe_oof")
    parser.add_argument("--regime_oof", default="/private/tmp/regime_stack_2024_oof.npz")
    parser.add_argument("--output_dir", default="/private/tmp/clean_dsf2_crossed")
    args = parser.parse_args()
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    train = frame.loc[frame["season"].eq(2023)].copy()
    valid = frame.loc[frame["season"].eq(2024)].copy()
    train_oof = align(Path(args.clean_oof_dir) / "clean_moe_2023.npz", train["row_id"].astype(str).to_numpy())
    valid_oof = align(Path(args.clean_oof_dir) / "clean_moe_2024.npz", valid["row_id"].astype(str).to_numpy())
    regime = align(Path(args.regime_oof), valid["row_id"].astype(str).to_numpy())
    base = np.clip(0.70 * valid_oof["prediction"] + 0.30 * regime["prediction"] - 0.002, 0.0, 1.0)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=3, dtype=np.float32)
    x_train_cat = encoder.fit_transform(categorical(train))
    x_valid_cat = encoder.transform(categorical(valid))
    x_train_num, x_valid_num = numeric(train, valid)
    x_train = sparse.hstack([x_train_cat, x_train_num], format="csr")
    x_valid = sparse.hstack([x_valid_cat, x_valid_num], format="csr")
    print(f"matrix train={x_train.shape} nnz={x_train.nnz:,}", flush=True)
    y_train = train[TARGET].to_numpy(float)
    y = valid[TARGET].to_numpy(float)
    select = valid_oof["select"].astype(bool)
    evaluate = valid_oof["evaluate"].astype(bool)
    candidates = []
    residual_candidates = []
    for alpha in ALPHAS:
        direct = Ridge(alpha=alpha, solver="lsqr", tol=1e-6, max_iter=5000)
        direct.fit(x_train, y_train)
        raw = np.clip(direct.predict(x_valid), 0.0, 1.0)
        params, candidate = choose(base, raw, y, select)
        candidates.append((np.mean((y[select] - candidate[select]) ** 2), alpha, params, candidate, raw))
        residual = Ridge(alpha=alpha, solver="lsqr", tol=1e-6, max_iter=5000, fit_intercept=False)
        residual.fit(x_train, y_train - train_oof["prediction"])
        correction = residual.predict(x_valid)
        for strength in (0.05, 0.10, 0.20, 0.30, 0.50):
            for cap in (0.005, 0.010, 0.020, 0.030):
                adjusted = np.clip(base + strength * np.clip(correction, -cap, cap), 0.0, 1.0)
                residual_candidates.append((np.mean((y[select] - adjusted[select]) ** 2), alpha, strength, cap, adjusted, correction))
    best_direct = min(candidates, key=lambda item: item[0])
    best_residual = min(residual_candidates, key=lambda item: item[0])
    if best_residual[0] < best_direct[0]:
        _, alpha, strength, cap, candidate, correction = best_residual
        selection = {"mode": "residual", "alpha": alpha, "strength": strength, "cap": cap}
        alternative = correction
    else:
        _, alpha, params, candidate, alternative = best_direct
        selection = {"mode": "direct", "alpha": alpha, "blend": params}
    report = {
        "selection": selection,
        "matrix": {"rows": len(train), "columns": x_train.shape[1], "nnz": x_train.nnz},
        "select": {"base": bss(y, base, select), "candidate": bss(y, candidate, select)},
        "evaluate": {"base": bss(y, base, evaluate), "candidate": bss(y, candidate, evaluate)},
        "segments": {},
        "correlation": float(np.corrcoef(y[evaluate] - base[evaluate], alternative[evaluate])[0, 1]),
        "rules": "2023 official train crossed effects; tune 2024 select; untouched 2024 evaluate; row-local frozen encoder",
    }
    for game_type in ("F", "R"):
        mask = evaluate & valid["game_type"].astype(str).eq(game_type).to_numpy()
        report["segments"][game_type] = {"rows": int(mask.sum()), "base": bss(y, base, mask), "candidate": bss(y, candidate, mask)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", row_id=valid["row_id"].astype(str), y=y, base=base, candidate=candidate, select=select, evaluate=evaluate)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
