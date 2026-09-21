#!/usr/bin/env python3
"""OOF-frozen blend of probe-free team DSF and clean_lookup."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = "./data"
MODEL_DIR = "./model"
OUT_PATH = "./output/submission.csv"
ID = "row_id"
TARGET = "control_success"
DSF_WEIGHT = 0.18
SHIFT = -0.002


def run_member(root: Path, name: str, output: Path) -> pd.DataFrame:
    code = """
import importlib.util
import sys
from pathlib import Path
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('_clean_dsf2_child', root / 'script.py')
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


def aligned(frame: pd.DataFrame, ids: pd.Series) -> np.ndarray:
    values = pd.Series(frame[TARGET].to_numpy(float), index=frame[ID].astype(str)).reindex(ids.astype(str)).to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("member alignment failed")
    return values


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(OUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")
    clean = aligned(run_member(root / "clean", "clean", output.parent / "_clean.csv"), sample[ID])
    dsf = aligned(run_member(root / "dsf", "dsf", output.parent / "_dsf.csv"), sample[ID])
    prediction = np.clip(DSF_WEIGHT * dsf + (1.0 - DSF_WEIGHT) * clean + SHIFT, 0.0, 1.0)
    sample[TARGET] = prediction
    sample.to_csv(output, index=False)
    print(f"clean dsf2 rows={len(sample)} dsf_weight={DSF_WEIGHT:.2f}")


if __name__ == "__main__":
    main()
