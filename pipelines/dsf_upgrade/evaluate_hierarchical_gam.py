#!/usr/bin/env python3
"""Walk-forward OOF evaluation for the sparse hierarchical logistic GAM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from pipelines.dsf_upgrade.hierarchical_gam import FrozenTransform


def brier(y, p): return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))
def bss(y, p):
    y = np.asarray(y, float); return 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean())))


def train_epoch(model, transform, frame, y, sample_weight, order, chunk=100000):
    for start in range(0, len(order), chunk):
        idx = order[start:start + chunk]
        matrix = transform.transform(frame.iloc[idx])
        model.partial_fit(matrix, y[idx], classes=np.array([0, 1]), sample_weight=sample_weight[idx])


def predict(model, transform, frame, chunk=100000):
    result = []
    for start in range(0, len(frame), chunk):
        matrix = transform.transform(frame.iloc[start:start + chunk])
        result.append(model.predict_proba(matrix)[:, 1])
    return np.concatenate(result)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--dsf_oof", required=True); ap.add_argument("--out_dir", default="./hierarchical_gam_oof")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--alpha", type=float, default=1e-5)
    ap.add_argument("--eta0", type=float, default=1e-3); args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False); frame["row_id"] = frame["row_id"].astype(str)
    y = frame["control_success"].to_numpy(np.int8); seasons = frame["season"].to_numpy(int)
    with np.load(args.dsf_oof, allow_pickle=False) as saved:
        dsf = pd.DataFrame({"row_id": saved["row_id"].astype(str), "dsf": saved["blend_061"], "saved_y": saved["y"]})
    results = {}
    for year in (2023, 2024):
        train = frame.loc[seasons < year].reset_index(drop=True); target = y[seasons < year]
        valid = frame.loc[seasons == year].reset_index(drop=True); valid_y = y[seasons == year]
        hashes = pd.util.hash_pandas_object(valid["row_id"], index=False).to_numpy(); stop = hashes % 4 < 2; select = hashes % 4 == 2; evaluate = hashes % 4 == 3
        transform = FrozenTransform().fit(train)
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=args.alpha,
            learning_rate="constant", eta0=args.eta0, average=True, random_state=55000 + year)
        train_year = train["season"].to_numpy(int); weight = (1 + .10 * (train_year - train_year.min())).astype(np.float64); weight /= weight.mean()
        rng = np.random.default_rng(55000 + year)
        for epoch in range(args.epochs):
            train_epoch(model, transform, train, target, weight, rng.permutation(len(train)))
            stop_pred = predict(model, transform, valid.loc[stop])
            print(f"year={year} epoch={epoch+1} stop_brier={brier(valid_y[stop], stop_pred):.8f}", flush=True)
        gam = predict(model, transform, valid)
        local = pd.DataFrame({"row_id": valid["row_id"], "y": valid_y, "gam": gam, "stop": stop, "select": select, "evaluate": evaluate}).merge(dsf, on="row_id", validate="one_to_one")
        if not np.array_equal(local["y"].to_numpy(float), local["saved_y"].to_numpy(float)): raise RuntimeError("target mismatch")
        sy = local["y"].to_numpy(float); a = local["dsf"].to_numpy(float); g = local["gam"].to_numpy(float)
        sm = local["select"].to_numpy(bool); em = local["evaluate"].to_numpy(bool)
        grid = np.arange(0, .301, .01); chosen = min(grid, key=lambda w: brier(sy[sm], (1-w)*a[sm] + w*g[sm])); blend = (1-chosen)*a + chosen*g
        result = {"weight": float(chosen), "dsf_bss": bss(sy[em], a[em]), "gam_bss": bss(sy[em], g[em]), "blend_bss": bss(sy[em], blend[em]), "delta_bss": bss(sy[em], blend[em])-bss(sy[em], a[em]), "residual_correlation": float(np.corrcoef(a[em]-sy[em], g[em]-sy[em])[0,1])}
        results[str(year)] = result; print(year, json.dumps(result), flush=True)
        np.savez_compressed(out / f"gam_{year}.npz", row_id=local["row_id"].to_numpy(str), y=sy.astype(np.float32), dsf=a.astype(np.float32), gam=g.astype(np.float32), select=sm, evaluate=em)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
