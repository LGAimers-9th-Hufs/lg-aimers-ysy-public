#!/usr/bin/env python3
"""Frozen DSF + FactorTabM conditional router inference."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from pipelines.dsf_upgrade.factor_tabm import FactorTabM, make_features, predict


DATA_DIR, MODEL_DIR, OUT_PATH = "./data", "./model", "./output/submission.csv"
ID, TARGET = "row_id", "control_success"


def _router_features(dsf, factor):
    diff = factor-dsf
    return np.column_stack([dsf, factor, diff, np.abs(diff), np.abs(dsf-.5), np.abs(factor-.5), dsf*(1-dsf)])


def main():
    root = Path(__file__).resolve().parent; output = Path(OUT_PATH); output.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(DATA_DIR)/"test.csv", encoding="utf-8-sig", low_memory=False); sample = pd.read_csv(Path(DATA_DIR)/"sample_submission.csv", encoding="utf-8-sig")
    dsf_root = root/"dsf"; spec = importlib.util.spec_from_file_location("_factor_router_dsf", dsf_root/"member_script.py"); module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(dsf_root))
    try:
        spec.loader.exec_module(module); module.DATA_DIR=DATA_DIR; module.MODEL_DIR=str(dsf_root/"model"); module.OUT_PATH=str(output.parent/"_dsf.csv"); module.main()
    finally: sys.path.pop(0)
    raw = pd.read_csv(output.parent/"_dsf.csv"); dsf = pd.Series(raw[TARGET].to_numpy(float), index=raw[ID].astype(str)).reindex(test[ID].astype(str)).to_numpy()
    artifact = torch.load(Path(MODEL_DIR)/"factor_tabm.pt", map_location="cpu", weights_only=True); pre = joblib.load(Path(MODEL_DIR)/"preprocessor.joblib")
    cats, nums = pre.transform(make_features(test)); model = FactorTabM(artifact["cardinalities"], artifact["n_num"], artifact["pair_indices"], k=artifact["k"]); model.load_state_dict(artifact["state_dict"])
    factor = predict(model, cats, nums); router = joblib.load(Path(MODEL_DIR)/"router.joblib"); route = router.predict_proba(_router_features(dsf, factor))[:,1]
    meta = json.loads((Path(MODEL_DIR)/"metadata.json").read_text()); use = route >= float(meta["threshold"]); prediction = dsf + use*float(meta["alpha"])*(factor-dsf)
    aligned = pd.Series(np.clip(prediction,0,1), index=test[ID].astype(str)).reindex(sample[ID].astype(str)).to_numpy()
    if not np.isfinite(aligned).all(): raise RuntimeError("row alignment failed")
    sample[TARGET]=aligned; sample.to_csv(output,index=False); print(f"DSF/FactorTabM router rows={len(sample)} routed={int(use.sum())}")


if __name__ == "__main__": main()
