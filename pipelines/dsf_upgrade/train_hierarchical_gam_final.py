#!/usr/bin/env python3
"""Fit the final train-frozen hierarchical GAM artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from pipelines.dsf_upgrade.evaluate_hierarchical_gam import train_epoch
from pipelines.dsf_upgrade.hierarchical_gam import FrozenTransform


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--output_dir", default="./hierarchical_gam_final"); ap.add_argument("--seed", type=int, default=57025)
    args = ap.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    y = frame["control_success"].to_numpy(np.int8); transform = FrozenTransform().fit(frame)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-5, learning_rate="constant",
        eta0=2e-4, average=True, random_state=args.seed)
    years = frame["season"].to_numpy(int); weight = (1 + .10 * (years - years.min())).astype(float); weight /= weight.mean()
    order = np.random.default_rng(args.seed).permutation(len(frame))
    train_epoch(model, transform, frame, y, weight, order)
    joblib.dump({"transform": transform, "model": model}, output / "hierarchical_gam.joblib", compress=3)
    (output / "metadata.json").write_text(json.dumps({
        "train_rows": len(frame), "max_season": int(years.max()), "epochs": 1,
        "alpha": 5e-5, "eta0": 2e-4, "rules": "train-frozen row-local transform"
    }, indent=2), encoding="utf-8")
    print(f"saved {output.resolve()}")


if __name__ == "__main__": main()
