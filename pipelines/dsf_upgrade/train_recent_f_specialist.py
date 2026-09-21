#!/usr/bin/env python3
"""Train a 2023+ recent-regime F CatBoost specialist around frozen DSF OOF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize

from shared.catboost_features import ALL_CAT_COLS, ID_COL, TARGET_COL, build_features


def brier(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def bss(y, p):
    y = np.asarray(y, float)
    return 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean())))


def beta_fit(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.column_stack([np.log(p), -np.log1p(-p), np.ones(len(p))])
    def objective(coef):
        q = 1.0 / (1.0 + np.exp(-np.clip(x @ coef, -30, 30)))
        return brier(y, q) + 1e-4 * ((coef[0] - 1) ** 2 + (coef[1] - 1) ** 2 + coef[2] ** 2)
    fit = minimize(objective, [1.0, 1.0, 0.0], method="L-BFGS-B")
    return np.asarray(fit.x, float)


def beta_apply(p, coef):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = coef[0] * np.log(p) - coef[1] * np.log1p(-p) + coef[2]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def params(seed, iterations):
    return dict(iterations=iterations, depth=6, learning_rate=0.035, loss_function="Logloss",
        eval_metric="BrierScore", random_seed=seed, l2_leaf_reg=14.0, random_strength=0.8,
        bootstrap_type="Bernoulli", subsample=0.82, rsm=0.82, boosting_type="Ordered",
        one_hot_max_size=16, max_ctr_complexity=1, thread_count=6, verbose=100,
        allow_writing_files=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--dsf_oof", required=True)
    ap.add_argument("--out_dir", default="./model_dsf_recent_f")
    ap.add_argument("--report", default="./dsf_recent_f_report.json")
    ap.add_argument("--game_type", choices=["F", "R"], default="F")
    args = ap.parse_args()
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    frame[ID_COL] = frame[ID_COL].astype(str)
    with np.load(args.dsf_oof, allow_pickle=False) as saved:
        oof = pd.DataFrame({ID_COL: saved["row_id"].astype(str), "dsf": saved["blend_061"].astype(float), "saved_y": saved["y"].astype(float)})
    aligned = frame[[ID_COL, "season", "game_type", TARGET_COL]].merge(oof, on=ID_COL, how="inner", validate="one_to_one")
    if not np.array_equal(aligned[TARGET_COL].to_numpy(float), aligned.saved_y.to_numpy(float)):
        raise RuntimeError("DSF OOF target mismatch")
    regime = args.game_type
    train_mask = (frame.season == 2023) & (frame.game_type.astype(str) == regime)
    valid_oof = aligned[(aligned.season == 2024) & (aligned.game_type.astype(str) == regime)].copy()
    valid = frame[frame[ID_COL].isin(valid_oof[ID_COL])].copy()
    valid = valid.set_index(ID_COL).loc[valid_oof[ID_COL]].reset_index()
    x_train = build_features(frame.loc[train_mask]); y_train = frame.loc[train_mask, TARGET_COL].to_numpy(np.int8)
    x_valid = build_features(valid); y_valid = valid[TARGET_COL].to_numpy(np.int8); dsf = valid_oof.dsf.to_numpy(float)
    cats = [c for c in ALL_CAT_COLS if c in x_train]; cat_idx = [x_train.columns.get_loc(c) for c in cats]
    hashes = pd.util.hash_pandas_object(valid[ID_COL].astype(str), index=False).to_numpy()
    stop, select, evaluate = (hashes % 4) < 2, (hashes % 4) == 2, (hashes % 4) == 3
    model = CatBoostClassifier(**params(26031, 1200))
    model.fit(Pool(x_train, y_train, cat_features=cat_idx),
        eval_set=Pool(x_valid.loc[stop], y_valid[stop], cat_features=cat_idx),
        early_stopping_rounds=140, use_best_model=True)
    specialist = model.predict_proba(Pool(x_valid, cat_features=cat_idx))[:, 1]
    residual_corr = float(np.corrcoef(dsf[evaluate] - y_valid[evaluate], specialist[evaluate] - y_valid[evaluate])[0, 1])
    candidates = []
    for weight in np.arange(0.0, 0.201, 0.01):
        blend = (1.0 - weight) * dsf + weight * specialist
        candidates.append((brier(y_valid[select], blend[select]), float(weight)))
    weight = min(candidates)[1]
    blend = (1.0 - weight) * dsf + weight * specialist
    beta = beta_fit(y_valid[select], blend[select])
    dsf_beta = beta_fit(y_valid[select], dsf[select])
    raw_eval = bss(y_valid[evaluate], blend[evaluate])
    beta_eval = bss(y_valid[evaluate], beta_apply(blend[evaluate], beta))
    baseline_eval = bss(y_valid[evaluate], dsf[evaluate])
    baseline_beta_eval = bss(y_valid[evaluate], beta_apply(dsf[evaluate], dsf_beta))
    # Calibration is optional and must improve the untouched partition.
    use_beta = bool(beta_eval > raw_eval and beta_eval > baseline_eval and beta_eval > baseline_beta_eval)
    best = int(model.get_best_iteration() + 1)
    deploy_mask = frame.season.isin([2023, 2024]) & (frame.game_type.astype(str) == regime)
    x_deploy = build_features(frame.loc[deploy_mask]); y_deploy = frame.loc[deploy_mask, TARGET_COL].to_numpy(np.int8)
    final = CatBoostClassifier(**params(26031, best)); final.fit(Pool(x_deploy, y_deploy, cat_features=cat_idx))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True); final.save_model(out / "recent_specialist.cbm")
    artifact = {"version": 1, "feature_cols": x_train.columns.tolist(), "cat_cols": cats,
        "game_type": regime, "blend_weight": weight, "beta": beta.tolist(), "use_beta": use_beta,
        "validation": {"train": f"2023 {regime}", "valid": f"2024 {regime} hash split", "best_iteration": best,
            "baseline_bss": baseline_eval, "baseline_beta_bss": baseline_beta_eval,
            "blend_raw_bss": raw_eval, "blend_beta_bss": beta_eval,
            "residual_correlation": residual_corr, "n_stop": int(stop.sum()), "n_select": int(select.sum()), "n_eval": int(evaluate.sum())},
        "deployment": {"train": f"2023-2024 {regime}", "rows": int(deploy_mask.sum())},
        "rules": "official train and DSF train OOF only; no test/leaderboard fitting"}
    (out / "artifacts.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.report).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
