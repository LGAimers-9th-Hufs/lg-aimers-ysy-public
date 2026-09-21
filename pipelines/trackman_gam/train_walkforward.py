#!/usr/bin/env python3
"""ID-free TrackMan spline GAM with strict prior-season physical artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.trackman.command_features import enrich_command  # noqa: E402

TARGET = "control_success"
BASE_NUMERIC = [
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "num_runners_on",
    "home_win_expectancy", "li", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
CATEGORICAL = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "count", "count_hand", "regime_count", "regime_base", "inning_runner",
]


def prepare(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in BASE_NUMERIC:
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    tm = enrich_command(frame, artifact)
    # Compact physical subset: stability, movement, release and year-over-year drift.
    selected = [
        "tmcmd_release_side_std", "tmcmd_release_height_std", "tmcmd_release_ellipse",
        "tmcmd_speed_std", "tmcmd_spin_std", "tmcmd_extension_std",
        "tmcmd_movement_ellipse", "tmcmd_active_spin_proxy_std",
        "tmcmd_velo_retention_std", "tmcmd_drift_release", "tmcmd_drift_speed",
        "tmcmd_drift_movement", "tmcmd_release_risk", "tmcmd_movement_risk",
    ]
    for column in selected:
        out[column] = tm[column].to_numpy()
    count = frame["balls_before"].astype("string") + "-" + frame["strikes_before"].astype("string")
    hand = frame["pitcher_hand"].astype("string") + "-" + frame["batter_hand"].astype("string")
    out["top_bottom"] = frame["top_bottom"].astype("string")
    out["game_type"] = frame["game_type"].astype("string")
    out["base_state"] = frame["base_state"].astype("string")
    out["pitcher_hand"] = frame["pitcher_hand"].astype("string")
    out["batter_hand"] = frame["batter_hand"].astype("string")
    out["count"] = count
    out["count_hand"] = count + "|" + hand
    out["regime_count"] = frame["game_type"].astype("string") + "|" + count
    out["regime_base"] = frame["game_type"].astype("string") + "|" + frame["base_state"].astype("string")
    out["inning_runner"] = frame["inning"].astype("string") + "|" + frame["num_runners_on"].astype("string")
    return out


def bss(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    yy = y[mask]
    return float(100000 * (1 - np.mean((yy - prediction[mask]) ** 2) / (yy.mean() * (1 - yy.mean()))))


def calibrate(y: np.ndarray, prediction: np.ndarray, select: np.ndarray) -> tuple[np.ndarray, dict]:
    a, b = np.linalg.lstsq(
        np.column_stack([np.ones(select.sum()), prediction[select] - 0.5]), y[select] - 0.5, rcond=None
    )[0]
    return np.clip(0.5 + a + b * (prediction - 0.5), 0, 1), {"intercept": float(a), "slope": float(b)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--artifact", type=Path, default=ROOT / "trackman_command_artifacts.json")
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/trackman_gam"))
    args = parser.parse_args()
    frame = pd.read_csv(args.train, encoding="utf-8-sig", low_memory=False)
    train = frame.loc[frame["season"].eq(2023)].reset_index(drop=True)
    valid = frame.loc[frame["season"].eq(2024)].reset_index(drop=True)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    x_train = prepare(train, artifact)
    x_valid = prepare(valid, artifact)
    numeric_columns = [c for c in x_train.columns if c not in CATEGORICAL]
    transformer = ColumnTransformer([
        ("spline", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False, sparse_output=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=20, dtype=np.float32), CATEGORICAL),
    ], sparse_threshold=1.0)
    matrix_train = transformer.fit_transform(x_train)
    matrix_valid = transformer.transform(x_valid)
    y_train = train[TARGET].to_numpy(float)
    y = valid[TARGET].to_numpy(float)
    split = np.load("/private/tmp/regime_stack_2024_oof.npz", allow_pickle=True)
    if not np.array_equal(valid["row_id"].astype(str).to_numpy(), split["row_id"].astype(str)):
        raise RuntimeError("validation alignment failed")
    select = split["select"]
    evaluate = split["evaluate"]
    candidates = []
    for alpha in (10.0, 30.0, 100.0, 300.0, 1000.0):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-6, max_iter=5000)
        model.fit(matrix_train, y_train)
        raw = np.clip(model.predict(matrix_valid), 0, 1)
        prediction, calibration = calibrate(y, raw, select)
        candidates.append((np.mean((y[select] - prediction[select]) ** 2), alpha, model, prediction, calibration))
        print(f"alpha={alpha:g} select={bss(y,prediction,select):.3f} evaluate={bss(y,prediction,evaluate):.3f}", flush=True)
    _, alpha, model, prediction, calibration = min(candidates, key=lambda item: item[0])
    report = {
        "alpha": alpha, "calibration": calibration,
        "standalone": {"select": bss(y, prediction, select), "evaluate": bss(y, prediction, evaluate)},
        "matrix": {"train": list(matrix_train.shape), "valid": list(matrix_valid.shape)},
        "rules": "2023 official train + prior-season official TrackMan artifacts only -> 2024 OOF; no player IDs/test aggregation/LB feedback",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"transformer": transformer, "model": model, "calibration": calibration, "artifact": artifact}, args.output / "trackman_gam.joblib", compress=3)
    np.savez_compressed(args.output / "predictions.npz", row_id=valid["row_id"].astype(str).to_numpy(), y=y, prediction=prediction, select=select, evaluate=evaluate)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
