#!/usr/bin/env python3
"""과거 시점 walk-forward fold에서 공통으로 안정적인 피처를 선택하고 2024에서 평가한다.

피처 선택에는 2024 label을 사용하지 않는다. test는 읽지 않는다.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from shared.catboost_features import ALL_CAT_COLS, ID_COL, TARGET_COL, build_features
from pipelines.catboost.experiments.evaluate_catboost_blend import bss, calibrate, fit_cal, full_params, recent_params


def params(seed: int, iterations: int, verbose: int = 0):
    p = full_params(iterations)
    p.update(random_seed=seed, verbose=verbose, depth=7, rsm=0.9)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--out", default="./fold_stability.json")
    ap.add_argument("--n_features", type=int, default=45)
    a = ap.parse_args()
    started = time.time()

    df = pd.read_csv(a.train_path, encoding="utf-8-sig")
    x = build_features(df)
    y = df[TARGET_COL].to_numpy(np.int8)
    seasons = df["season"].to_numpy()
    cats_all = [c for c in ALL_CAT_COLS if c in x]

    # Selection folds end in 2023: 2024 is completely untouched by selection.
    fold_importance = {}
    for val_year in (2021, 2022, 2023):
        tr = seasons < val_year
        va = seasons == val_year
        cat_idx = [x.columns.get_loc(c) for c in cats_all]
        model = CatBoostClassifier(**params(5000 + val_year, 220))
        model.fit(
            Pool(x.loc[tr], y[tr], cat_features=cat_idx),
            eval_set=Pool(x.loc[va], y[va], cat_features=cat_idx),
            verbose=False,
        )
        imp = model.get_feature_importance(type="PredictionValuesChange")
        fold_importance[str(val_year)] = dict(zip(x.columns, map(float, imp)))

    imp_df = pd.DataFrame(fold_importance).fillna(0.0)
    # Fold 내 percentile로 스케일을 맞추고, 최악 fold를 크게 반영한다.
    pct = imp_df.rank(pct=True, axis=0)
    score = 0.55 * pct.mean(axis=1) + 0.45 * pct.min(axis=1) - 0.20 * pct.std(axis=1)
    stable = score.sort_values(ascending=False).head(min(a.n_features, len(score))).index.tolist()
    stable_set = set(stable)
    stable = [c for c in x.columns if c in stable_set]
    stable_cats = [c for c in cats_all if c in stable_set]
    xs = x[stable]
    cat_idx = [xs.columns.get_loc(c) for c in stable_cats]

    val = seasons == 2024
    train = seasons < 2024
    xv, yv = xs.loc[val], y[val]
    hashes = pd.util.hash_pandas_object(df.loc[val, ID_COL].astype(str), index=False).to_numpy()
    stop = (hashes % 4) < 2
    cal_fit = (hashes % 4) == 2
    final_eval = (hashes % 4) == 3

    fy = seasons[train]
    fw = (1 + .18 * (fy - 2019)).astype(np.float32)
    full = CatBoostClassifier(**full_params(1000))
    full.fit(
        Pool(xs.loc[train], y[train], cat_features=cat_idx, weight=fw),
        eval_set=Pool(xv.loc[stop], yv[stop], cat_features=cat_idx),
        early_stopping_rounds=120, use_best_model=True,
    )
    pf = full.predict_proba(Pool(xv, cat_features=cat_idx))[:, 1]

    recent = np.isin(seasons, [2021, 2022, 2023])
    ry = seasons[recent]
    rw = (1 + .15 * (ry - 2021)).astype(np.float32)
    rec = CatBoostClassifier(**recent_params(1000))
    rec.fit(
        Pool(xs.loc[recent], y[recent], cat_features=cat_idx, weight=rw),
        eval_set=Pool(xv.loc[stop], yv[stop], cat_features=cat_idx),
        early_stopping_rounds=120, use_best_model=True,
    )
    pr = rec.predict_proba(Pool(xv, cat_features=cat_idx))[:, 1]

    grid = np.linspace(0, 1, 201)
    losses = [np.mean(((w * pf + (1-w) * pr)[cal_fit] - yv[cal_fit]) ** 2) for w in grid]
    weight = float(grid[int(np.argmin(losses))])
    blend = weight * pf + (1-weight) * pr
    cal = fit_cal(yv[cal_fit], blend[cal_fit])
    result = {
        "selection_folds": [2021, 2022, 2023],
        "selection_note": "feature selection ends at 2023; 2024 labels excluded",
        "n_features": len(stable),
        "feature_cols": stable,
        "cat_cols": stable_cats,
        "stability_score": {c: float(score[c]) for c in stable},
        "fold_importance": fold_importance,
        "best_iterations": {"full": full.get_best_iteration()+1, "recent3": rec.get_best_iteration()+1},
        "weights": [weight, 1-weight],
        "calibration": cal,
        "full_eval": bss(yv[final_eval], pf[final_eval]),
        "recent_eval": bss(yv[final_eval], pr[final_eval]),
        "blend_cal_eval": bss(yv[final_eval], calibrate(blend[final_eval], cal)),
        "elapsed_seconds": time.time()-started,
        "rules": "official train walk-forward only; no test or leaderboard information",
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
