#!/usr/bin/env python3
"""Frozen OOF blend of probe-free calibrated DSF and clean lookup."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = "./data"
OUT_PATH = "./output/submission.csv"
ID = "row_id"
TARGET = "control_success"
LOOKUP_WEIGHT = 0.50
LOOKUP_INTERCEPT = 0.004027927737338711
LOOKUP_SLOPE = 1.1120487007912476


def run_member(root: Path, output: Path) -> pd.DataFrame:
    code = """
import importlib.util
import sys
from pathlib import Path
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('_dsf_lookup_child', root / 'script.py')
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(root))
spec.loader.exec_module(module)
module.DATA_DIR = sys.argv[2]
module.MODEL_DIR = str(root / 'model')
module.OUT_PATH = sys.argv[3]
module.main()
"""
    subprocess.run(
        [sys.executable, "-c", code, str(root), str(Path(DATA_DIR).resolve()), str(output.resolve())],
        check=True,
    )
    return pd.read_csv(output)


def aligned(frame: pd.DataFrame, ids: pd.Series, name: str) -> np.ndarray:
    if ID not in frame or TARGET not in frame:
        raise RuntimeError(f"{name} output columns are invalid")
    values = pd.Series(
        pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(np.float64),
        index=frame[ID].astype(str),
    ).reindex(ids.astype(str)).to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} output alignment failed")
    return values


def blend_predictions(dsf: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    calibrated_lookup = np.clip(
        0.5 + LOOKUP_INTERCEPT + LOOKUP_SLOPE * (lookup - 0.5), 0.0, 1.0
    )
    return np.clip((1.0 - LOOKUP_WEIGHT) * dsf + LOOKUP_WEIGHT * calibrated_lookup, 0.0, 1.0)


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(OUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")
    dsf = aligned(run_member(root / "dsf", output.parent / "_dsf.csv"), sample[ID], "dsf")
    lookup = aligned(run_member(root / "lookup", output.parent / "_lookup.csv"), sample[ID], "lookup")
    sample[TARGET] = blend_predictions(dsf, lookup)
    sample.to_csv(output, index=False)
    print(f"dsf lookup rows={len(sample)} lookup_weight={LOOKUP_WEIGHT:.2f}")


if __name__ == "__main__":
    main()
