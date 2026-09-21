#!/usr/bin/env python3
"""Fit the final residual model using 2024 strict OOF clean_lookup errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from pipelines.clean_rescat.features import TARGET, build_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="<repo>/data/train.csv")
    parser.add_argument("--lookup-oof", default="/private/tmp/clean_lookup_eval/predictions.npz")
    parser.add_argument("--output", default="./clean_rescat_model")
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()
    frame = pd.read_csv(args.train, encoding="utf-8-sig", low_memory=False)
    frame = frame.loc[frame["season"].eq(2024)].copy()
    ids = frame["row_id"].astype(str).to_numpy()
    with np.load(args.lookup_oof, allow_pickle=True) as saved:
        source = saved["row_id"].astype(str)
        position = pd.Series(np.arange(len(source)), index=source).reindex(ids).to_numpy()
        if not np.isfinite(position).all():
            raise RuntimeError("lookup OOF alignment failed")
        base = saved["candidate"][position.astype(int)].astype(float)
    residual = frame[TARGET].to_numpy(float) - base
    x, cats = build_features(frame)
    model = CatBoostRegressor(
        loss_function="RMSE", boosting_type="Ordered", iterations=args.iterations,
        depth=6, learning_rate=0.03, l2_leaf_reg=80.0,
        random_strength=0.8, bootstrap_type="Bayesian", bagging_temperature=1.0,
        random_seed=20261010, thread_count=6, verbose=100, allow_writing_files=False,
    )
    model.fit(x, residual, cat_features=cats)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / "residual.cbm")
    metadata = {
        "version": "clean-rescat-v1", "training_season": 2024,
        "iterations": args.iterations, "strength": 0.10, "cap": 0.03,
        "validation": {"base_bss": 859.6619220956226, "candidate_bss": 861.0540283815693},
        "provenance": "official train only; 2024 strict OOF clean_lookup residual; no DSF/LB/test fitting",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
