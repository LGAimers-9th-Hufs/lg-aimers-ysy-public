#!/usr/bin/env python3
"""Row-independent clean_regime plus frozen official-train residual lookups."""
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_clean_lookup_base", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("base module loader unavailable")
    spec.loader.exec_module(module)
    return module


def keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].astype("string").fillna("<NA>").agg("|".join, axis=1)


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(OUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(DATA_DIR) / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")
    base_root = root / "base"
    sys.path.insert(0, str(base_root))
    try:
        base_module = load_module(base_root / "script.py")
        base_module.DATA_DIR = DATA_DIR
        base_module.MODEL_DIR = str(base_root / "model")
        base_module.OUT_PATH = str(output.parent / "_base.csv")
        base_module.main()
    finally:
        sys.path.pop(0)
    base_frame = pd.read_csv(output.parent / "_base.csv")
    base = pd.Series(base_frame[TARGET].to_numpy(float), index=base_frame[ID].astype(str)).reindex(test[ID].astype(str)).to_numpy()
    artifact = json.loads((root / "lookup.json").read_text(encoding="utf-8"))
    candidates = []
    for name, stats in artifact["statistics"].items():
        row_keys = keys(test, stats["columns"])
        sums = row_keys.map(stats["sum"]).fillna(0.0).to_numpy(float)
        counts = row_keys.map(stats["count"]).fillna(0.0).to_numpy(float)
        for kappa in artifact["kappas"]:
            candidates.append(sums / (counts + float(kappa)))
    matrix = np.column_stack(candidates)
    scale = np.asarray(artifact["scale"], dtype=float)
    coef = np.asarray(artifact["coef"], dtype=float)
    raw = (matrix / scale) @ coef
    prediction = np.clip(
        base + float(artifact["strength"]) * np.clip(raw, -float(artifact["cap"]), float(artifact["cap"])),
        0.0, 1.0,
    )
    aligned = pd.Series(prediction, index=test[ID].astype(str)).reindex(sample[ID].astype(str)).to_numpy()
    if not np.isfinite(aligned).all():
        raise RuntimeError("prediction alignment failed")
    sample[TARGET] = aligned
    sample.to_csv(output, index=False)
    print(f"clean lookup rows={len(sample)}")


if __name__ == "__main__":
    main()
