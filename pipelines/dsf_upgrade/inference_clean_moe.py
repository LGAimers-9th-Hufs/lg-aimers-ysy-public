#!/usr/bin/env python3
"""Clean Challenger + 3 FactorTabM + official TrackMan residual MoE."""
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
from pipelines.dsf_upgrade.residual_moe import make_moe_features
from pipelines.trackman.command_features import enrich_command


DATA_DIR, MODEL_DIR, OUT_PATH = "./data", "./model", "./output/submission.csv"
ID, TARGET = "row_id", "control_success"


def _load_factor(path: Path, preprocessor, features: pd.DataFrame) -> np.ndarray:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    cats, nums = preprocessor.transform(features)
    model = FactorTabM(
        artifact["cardinalities"], artifact["n_num"], artifact["pair_indices"], k=artifact["k"]
    )
    model.load_state_dict(artifact["state_dict"])
    return predict(model, cats, nums)


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(OUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(DATA_DIR) / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")
    challenger_root = root / "challenger"
    spec = importlib.util.spec_from_file_location("_clean_moe_challenger", challenger_root / "member_script.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(challenger_root))
    try:
        spec.loader.exec_module(module)
        module.DATA_DIR = DATA_DIR
        module.MODEL_DIR = str(challenger_root / "model")
        module.OUT_PATH = str(output.parent / "_challenger.csv")
        module.main()
    finally:
        sys.path.pop(0)
    raw = pd.read_csv(output.parent / "_challenger.csv")
    base = pd.Series(raw[TARGET].to_numpy(float), index=raw[ID].astype(str)).reindex(
        test[ID].astype(str)
    ).to_numpy()
    if not np.isfinite(base).all():
        raise RuntimeError("Challenger alignment failed")

    model_dir = Path(MODEL_DIR)
    base_features = make_features(test)
    factors = []
    for index in range(3):
        factors.append(
            _load_factor(
                model_dir / f"factor_{index}.pt",
                joblib.load(model_dir / f"preprocessor_{index}.joblib"),
                base_features,
            )
        )
    factor_matrix = np.column_stack(factors)
    command = json.loads((model_dir / "trackman_command_artifacts.json").read_text())
    track_features = pd.concat(
        [base_features.reset_index(drop=True), enrich_command(test, command).reset_index(drop=True)], axis=1
    )
    track = _load_factor(
        model_dir / "track_factor.pt", joblib.load(model_dir / "track_preprocessor.joblib"), track_features
    )
    zeros = np.zeros(len(test), dtype=float)
    features = make_moe_features(test, base, factor_matrix, track, zeros, base, zeros)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    residual = joblib.load(model_dir / "clean_moe.joblib").predict(features)
    prediction = base + float(metadata["alpha"]) * np.clip(
        residual, -float(metadata["cap"]), float(metadata["cap"])
    )
    aligned = pd.Series(np.clip(prediction, 0.0, 1.0), index=test[ID].astype(str)).reindex(
        sample[ID].astype(str)
    ).to_numpy()
    if not np.isfinite(aligned).all():
        raise RuntimeError("output alignment failed")
    sample[TARGET] = aligned
    sample.to_csv(output, index=False)
    print(f"clean residual MoE rows={len(sample)}")


if __name__ == "__main__":
    main()
