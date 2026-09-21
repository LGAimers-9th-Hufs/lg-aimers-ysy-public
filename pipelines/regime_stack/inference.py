#!/usr/bin/env python3
"""Clean MoE plus clean regime stack, with frozen train-OOF blend."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = "./data"
MODEL_DIR = "./model"
OUT_PATH = "./output/submission.csv"
ID = "row_id"
TARGET = "control_success"
REGIME_WEIGHT = 0.30
FINAL_SHIFT = -0.002


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(OUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(DATA_DIR) / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")

    clean_root = root / "clean"
    sys.path.insert(0, str(clean_root))
    try:
        clean = _module(clean_root / "member_script.py", "_regime_stack_clean_member")
        clean.DATA_DIR = DATA_DIR
        clean.MODEL_DIR = str(clean_root / "model")
        clean.OUT_PATH = str(output.parent / "_clean.csv")
        clean.main()
    finally:
        sys.path.pop(0)
    print("clean component complete", flush=True)
    clean_frame = pd.read_csv(output.parent / "_clean.csv")
    clean_prediction = pd.Series(
        clean_frame[TARGET].to_numpy(float), index=clean_frame[ID].astype(str)
    ).reindex(test[ID].astype(str)).to_numpy()

    regime_root = root / "regime"
    import lightgbm as lgb

    features = _module(regime_root / "features.py", "_regime_stack_features")
    metadata = json.loads((regime_root / "model" / "metadata.json").read_text(encoding="utf-8"))
    print("regime metadata loaded", flush=True)
    component = {}
    for name in ("specialist", "state", "invariant"):
        columns = metadata["feature_names"][name]
        matrix = features.build_features(test, metadata, columns)
        booster = lgb.Booster(model_file=str(regime_root / "model" / f"{name}.txt"))
        component[name] = booster.predict(matrix, num_threads=4)
        print(f"regime component {name} complete", flush=True)
    mixture = metadata["mixture"]
    regime_prediction = np.clip(
        sum(mixture[name] * component[name] for name in ("specialist", "state", "invariant"))
        + float(mixture["shift"]), 0.0, 1.0,
    )
    prediction = np.clip(
        (1.0 - REGIME_WEIGHT) * clean_prediction + REGIME_WEIGHT * regime_prediction + FINAL_SHIFT,
        0.0, 1.0,
    )
    aligned = pd.Series(prediction, index=test[ID].astype(str)).reindex(sample[ID].astype(str)).to_numpy()
    if not np.isfinite(aligned).all():
        raise RuntimeError("prediction alignment failed")
    sample[TARGET] = aligned
    sample.to_csv(output, index=False)
    print(f"clean regime stack rows={len(sample)}")


if __name__ == "__main__":
    main()
