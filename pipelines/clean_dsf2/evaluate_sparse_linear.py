#!/usr/bin/env python3
"""Evaluate row-local hashed sparse interactions across the 2023->2024 boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDRegressor

from pipelines.clean_forest.evaluate_residual_lookup import align, bss


TARGET = "control_success"
ALPHAS = (1e-6, 3e-6, 1e-5, 3e-5)
CATEGORICAL = (
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "pitcher_id", "batter_id", "num_runners_on",
)
NUMERIC = (
    "li", "home_win_expectancy", "score_diff_pitcher_team",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)


def token_rows(frame: pd.DataFrame):
    values = {column: frame[column].astype("string").fillna("NA").to_numpy() for column in CATEGORICAL}
    nums = {column: pd.to_numeric(frame[column], errors="coerce").to_numpy(float) for column in NUMERIC}
    n = len(frame)
    for i in range(n):
        row = {f"c:{column}={values[column][i]}": 1.0 for column in CATEGORICAL}
        count = f"{values['balls_before'][i]}-{values['strikes_before'][i]}"
        hand = f"{values['pitcher_hand'][i]}-{values['batter_hand'][i]}"
        context = f"{count}-{values['base_state'][i]}"
        interactions = (
            f"count_hand={count}|{hand}",
            f"count_base={context}",
            f"inning_runners={values['inning'][i]}|{values['num_runners_on'][i]}",
            f"regime_count={values['game_type'][i]}|{count}",
            f"pitcher_count={values['pitcher_id'][i]}|{count}",
            f"pitcher_bhand={values['pitcher_id'][i]}|{values['batter_hand'][i]}",
            f"batter_count={values['batter_id'][i]}|{count}",
            f"pitcher_regime={values['pitcher_id'][i]}|{values['game_type'][i]}",
            f"pteam_count={values['pitcher_team_id'][i]}|{count}",
            f"bteam_count={values['batter_team_id'][i]}|{count}",
        )
        row.update({f"x:{token}": 1.0 for token in interactions})
        for column, array in nums.items():
            value = array[i]
            if not np.isfinite(value):
                row[f"missing:{column}"] = 1.0
                continue
            if column.endswith("_n"):
                transformed = np.log1p(max(value, 0.0)) / 10.0
                bucket = min(int(np.log1p(max(value, 0.0)) * 2.0), 20)
            elif "expectancy" in column:
                transformed = (value - 50.0) / 50.0
                bucket = int(np.clip(value // 5.0, 0, 20))
            elif column in {"li", "score_diff_pitcher_team"}:
                transformed = float(np.clip(value, -10.0, 10.0)) / 10.0
                bucket = int(np.clip(np.floor(value), -10, 10))
            else:
                transformed = value - 0.5
                bucket = int(np.clip(np.floor(value * 20.0), 0, 20))
            row[f"n:{column}"] = float(transformed)
            row[f"b:{column}={bucket}"] = 1.0
        yield row


def choose(base: np.ndarray, prediction: np.ndarray, y: np.ndarray, select: np.ndarray) -> tuple[dict, np.ndarray]:
    # Affine parameters and blend are fitted only on the designated 2024 select rows.
    centered = prediction[select] - 0.5
    design = np.column_stack([np.ones(select.sum()), centered])
    intercept, slope = np.linalg.lstsq(design, y[select] - 0.5, rcond=None)[0]
    calibrated = np.clip(0.5 + intercept + slope * (prediction - 0.5), 0.0, 1.0)
    best = None
    for weight in np.arange(0.0, 0.801, 0.025):
        for shift in np.arange(-0.004, 0.0041, 0.001):
            candidate = np.clip((1.0 - weight) * base + weight * calibrated + shift, 0.0, 1.0)
            loss = np.mean((y[select] - candidate[select]) ** 2)
            if best is None or loss < best[0]:
                best = (loss, weight, shift, candidate)
    return {"intercept": float(intercept), "slope": float(slope), "weight": float(best[1]), "shift": float(best[2])}, best[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--clean_oof_dir", default="/private/tmp/clean_moe_oof")
    parser.add_argument("--regime_oof", default="/private/tmp/regime_stack_2024_oof.npz")
    parser.add_argument("--output_dir", default="/private/tmp/clean_dsf2_sparse")
    parser.add_argument("--hash_bits", type=int, default=19)
    args = parser.parse_args()
    usecols = list(dict.fromkeys(["row_id", "season", TARGET, *CATEGORICAL, *NUMERIC]))
    frame = pd.read_csv(args.train_path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    train = frame.loc[frame["season"].eq(2023)].copy()
    valid = frame.loc[frame["season"].eq(2024)].copy()
    train_oof = align(Path(args.clean_oof_dir) / "clean_moe_2023.npz", train["row_id"].astype(str).to_numpy())
    valid_oof = align(Path(args.clean_oof_dir) / "clean_moe_2024.npz", valid["row_id"].astype(str).to_numpy())
    regime_oof = align(Path(args.regime_oof), valid["row_id"].astype(str).to_numpy())
    base = np.clip(0.70 * valid_oof["prediction"] + 0.30 * regime_oof["prediction"] - 0.002, 0.0, 1.0)
    hasher = FeatureHasher(n_features=1 << args.hash_bits, input_type="dict", alternate_sign=True, dtype=np.float32)
    print(f"hash train rows={len(train):,}", flush=True)
    x_train = hasher.transform(token_rows(train))
    print(f"hash valid rows={len(valid):,} nnz_train={x_train.nnz:,}", flush=True)
    x_valid = hasher.transform(token_rows(valid))
    y_train = train[TARGET].to_numpy(float)
    y = valid[TARGET].to_numpy(float)
    select = valid_oof["select"].astype(bool)
    evaluate = valid_oof["evaluate"].astype(bool)
    candidates = []
    predictions = {}
    residual_candidates = []
    for alpha in ALPHAS:
        model = SGDRegressor(
            loss="squared_error", penalty="l2", alpha=alpha, max_iter=40, tol=1e-5,
            learning_rate="adaptive", eta0=0.01, average=True, random_state=20260920,
        )
        model.fit(x_train, y_train)
        prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
        predictions[str(alpha)] = prediction
        params, candidate = choose(base, prediction, y, select)
        candidates.append((np.mean((y[select] - candidate[select]) ** 2), alpha, params, candidate))
        print(f"alpha={alpha} weight={params['weight']:.3f} select_bss={bss(y,candidate,select):.3f}", flush=True)
        residual_model = SGDRegressor(
            loss="squared_error", penalty="l2", alpha=alpha, max_iter=40, tol=1e-5,
            learning_rate="adaptive", eta0=0.01, average=True, random_state=20260921,
        )
        residual_model.fit(x_train, y_train - train_oof["prediction"])
        raw = residual_model.predict(x_valid)
        for strength in (0.05, 0.10, 0.20, 0.30, 0.50):
            for cap in (0.005, 0.010, 0.020, 0.030):
                for shift in (-0.002, -0.001, 0.0, 0.001, 0.002):
                    adjusted = np.clip(base + strength * np.clip(raw, -cap, cap) + shift, 0.0, 1.0)
                    loss = np.mean((y[select] - adjusted[select]) ** 2)
                    residual_candidates.append((loss, alpha, strength, cap, shift, adjusted, raw))
    _, alpha, params, candidate = min(candidates, key=lambda item: item[0])
    _, residual_alpha, residual_strength, residual_cap, residual_shift, residual_candidate, residual_raw = min(
        residual_candidates, key=lambda item: item[0]
    )
    if np.mean((y[select] - residual_candidate[select]) ** 2) < np.mean((y[select] - candidate[select]) ** 2):
        candidate = residual_candidate
        mode = "residual"
    else:
        mode = "direct"
    report = {
        "mode": mode, "alpha": alpha, "blend": params,
        "residual": {"alpha": residual_alpha, "strength": residual_strength, "cap": residual_cap, "shift": residual_shift},
        "select": {"base": bss(y, base, select), "candidate": bss(y, candidate, select)},
        "evaluate": {"base": bss(y, base, evaluate), "candidate": bss(y, candidate, evaluate)},
        "segments": {},
        "direct_residual_correlation": float(np.corrcoef(y[evaluate] - base[evaluate], y[evaluate] - predictions[str(alpha)][evaluate])[0, 1]),
        "correction_correlation": float(np.corrcoef(y[evaluate] - base[evaluate], residual_raw[evaluate])[0, 1]),
        "rules": "fit 2023 official train only; tune 2024 select; untouched 2024 evaluate; fixed row-local hashing",
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
