#!/usr/bin/env python3
"""Row-aligned DSF + hierarchical GAM inference."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_DIR, MODEL_DIR, OUT_PATH = "./data", "./model", "./output/submission.csv"
ID, TARGET = "row_id", "control_success"


def main():
    root = Path(__file__).resolve().parent; output = Path(OUT_PATH); output.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(DATA_DIR) / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")
    dsf_root = root / "dsf"; spec = importlib.util.spec_from_file_location("_frozen_dsf", dsf_root / "member_script.py")
    module = importlib.util.module_from_spec(spec); sys.path.insert(0, str(dsf_root))
    try:
        spec.loader.exec_module(module); module.DATA_DIR = DATA_DIR; module.MODEL_DIR = str(dsf_root / "model")
        module.OUT_PATH = str(output.parent / "_dsf.csv"); module.main()
    finally: sys.path.pop(0)
    dsf_frame = pd.read_csv(output.parent / "_dsf.csv"); dsf = pd.Series(dsf_frame[TARGET].to_numpy(float), index=dsf_frame[ID].astype(str)).reindex(sample[ID].astype(str)).to_numpy()
    artifact = joblib.load(Path(MODEL_DIR) / "hierarchical_gam.joblib")
    gam = artifact["model"].predict_proba(artifact["transform"].transform(test))[:, 1]
    gam = pd.Series(gam, index=test[ID].astype(str)).reindex(sample[ID].astype(str)).to_numpy()
    weight = json.loads((Path(MODEL_DIR) / "metadata.json").read_text())["gam_weight"]
    if not np.isfinite(dsf).all() or not np.isfinite(gam).all(): raise RuntimeError("row alignment failed")
    sample[TARGET] = np.clip((1-weight)*dsf + weight*gam, 0, 1); sample.to_csv(output, index=False)
    print(f"DSF/GAM rows={len(sample)} gam_weight={weight:.3f}")


if __name__ == "__main__": main()
