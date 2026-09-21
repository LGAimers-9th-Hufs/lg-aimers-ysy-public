#!/usr/bin/env python3
"""Row-independent clean_lookup plus frozen residual CatBoost."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

DATA_DIR = "./data"
OUT_PATH = "./output/submission.csv"
ID = "row_id"
TARGET = "control_success"


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(OUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(DATA_DIR) / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")
    member_root = root / "clean_lookup"
    spec = importlib.util.spec_from_file_location("_clean_rescat_base", member_root / "script.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("clean_lookup loader unavailable")
    member = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(member_root))
    try:
        spec.loader.exec_module(member)
        member.DATA_DIR = DATA_DIR
        member.OUT_PATH = str(output.parent / "_clean_lookup.csv")
        member.main()
    finally:
        sys.path.pop(0)
    base_frame = pd.read_csv(output.parent / "_clean_lookup.csv")
    base = pd.Series(base_frame[TARGET].to_numpy(float), index=base_frame[ID].astype(str)).reindex(test[ID].astype(str)).to_numpy()
    metadata = json.loads((root / "model" / "metadata.json").read_text(encoding="utf-8"))
    feature_spec = importlib.util.spec_from_file_location(
        "_clean_rescat_features", root / "pipelines" / "clean_rescat" / "features.py"
    )
    if feature_spec is None or feature_spec.loader is None:
        raise RuntimeError("residual feature loader unavailable")
    feature_module = importlib.util.module_from_spec(feature_spec)
    feature_spec.loader.exec_module(feature_module)
    x, cats = feature_module.build_features(test)
    model = CatBoostRegressor()
    model.load_model(root / "model" / "residual.cbm")
    correction = model.predict(x, thread_count=6)
    prediction = np.clip(base + float(metadata["strength"]) * np.clip(correction, -float(metadata["cap"]), float(metadata["cap"])), 0.0, 1.0)
    aligned = pd.Series(prediction, index=test[ID].astype(str)).reindex(sample[ID].astype(str)).to_numpy()
    if not np.isfinite(aligned).all():
        raise RuntimeError("row alignment failed")
    sample[TARGET] = aligned
    sample.to_csv(output, index=False)
    print(f"clean residual CatBoost rows={len(sample)}")


if __name__ == "__main__":
    main()
