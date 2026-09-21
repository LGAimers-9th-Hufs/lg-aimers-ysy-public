#!/usr/bin/env python3
"""Row-independent DSF blend with train-OOF-selected regime weights."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = "./data"
MODEL_DIR = "./model"
OUT_PATH = "./output/submission.csv"
ID_COL = "row_id"
TARGET_COL = "control_success"


def run_member(root: Path, name: str, output: Path) -> pd.DataFrame:
    member = root / name
    spec = importlib.util.spec_from_file_location(f"_regime_{name}", member / "member_script.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(member))
    try:
        spec.loader.exec_module(module)
        module.DATA_DIR = DATA_DIR
        module.MODEL_DIR = str(member / "model")
        module.OUT_PATH = str(output)
        module.main()
    finally:
        sys.path.pop(0)
    return pd.read_csv(output, encoding="utf-8-sig")


def align(frame: pd.DataFrame, ids: pd.Series, name: str) -> np.ndarray:
    if frame[ID_COL].isna().any() or not frame[ID_COL].is_unique:
        raise RuntimeError(f"invalid {name} row_id")
    values = pd.Series(pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(float), index=frame[ID_COL].astype(str)).reindex(ids.astype(str)).to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"failed to align {name}")
    return values


def beta_calibrate(values: np.ndarray, coef: list[float]) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    a, b, c = map(float, coef)
    z = a * np.log(p) - b * np.log1p(-p) + c
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def main() -> None:
    root = Path(__file__).resolve().parent
    meta = json.loads(Path(MODEL_DIR, "metadata.json").read_text(encoding="utf-8"))
    test = pd.read_csv(Path(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sample = pd.read_csv(Path(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    if len(test) != len(sample) or test[ID_COL].isna().any() or not test[ID_COL].is_unique:
        raise RuntimeError("invalid evaluation rows")
    out_dir = Path(OUT_PATH).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    champion = align(run_member(root, "champion", out_dir / "_champion.csv"), sample[ID_COL], "champion")
    challenger = align(run_member(root, "challenger", out_dir / "_challenger.csv"), sample[ID_COL], "challenger")
    by_id = pd.Series(test["game_type"].astype(str).to_numpy(), index=test[ID_COL].astype(str))
    regimes = sample[ID_COL].astype(str).map(by_id)
    if regimes.isna().any():
        raise RuntimeError("failed to align game_type")
    weights = regimes.map(meta["challenger_weight_by_game_type"]).fillna(float(meta["fallback_weight"])).to_numpy(float)
    shifts = regimes.map(meta.get("probability_shift_by_game_type", {})).fillna(0.0).to_numpy(float)
    prediction = np.clip((1.0 - weights) * champion + weights * challenger + shifts, 0.0, 1.0)
    for regime, coef in meta.get("beta_calibration_by_game_type", {}).items():
        mask = regimes.to_numpy(str) == str(regime)
        prediction[mask] = beta_calibrate(prediction[mask], coef)
    result = sample.copy()
    result[TARGET_COL] = prediction
    result.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"DSF regime blend rows={len(result)}")


if __name__ == "__main__":
    main()
