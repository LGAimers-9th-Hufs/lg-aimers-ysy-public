#!/usr/bin/env python3
"""DSF + seed FactorTabM + TrackMan safety gate + frozen residual correction."""
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
from pipelines.trackman.command_features import enrich_command


DATA_DIR, MODEL_DIR, OUT_PATH = "./data", "./model", "./output/submission.csv"
ID, TARGET = "row_id", "control_success"


def _load_predict(path: Path, preprocessor, features: pd.DataFrame) -> np.ndarray:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    cats, nums = preprocessor.transform(features)
    model = FactorTabM(
        artifact["cardinalities"], artifact["n_num"], artifact["pair_indices"], k=artifact["k"]
    )
    model.load_state_dict(artifact["state_dict"])
    return predict(model, cats, nums)


def _router_features(dsf: np.ndarray, factor: np.ndarray) -> np.ndarray:
    diff = factor - dsf
    return np.column_stack(
        [dsf, factor, diff, np.abs(diff), np.abs(dsf - 0.5), np.abs(factor - 0.5), dsf * (1.0 - dsf)]
    )


def _residual_features(
    dsf: np.ndarray,
    factor: np.ndarray,
    track: np.ndarray,
    route_probability: np.ndarray,
    routed: np.ndarray,
    unsafe: np.ndarray,
) -> np.ndarray:
    factor_delta = factor - dsf
    track_delta = track - dsf
    return np.column_stack(
        [
            dsf,
            factor,
            track,
            factor_delta,
            track_delta,
            np.abs(factor_delta),
            np.abs(track_delta),
            route_probability,
            routed.astype(float),
            unsafe.astype(float),
            factor_delta * track_delta,
            np.abs(dsf - 0.5),
        ]
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(OUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(DATA_DIR) / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", encoding="utf-8-sig")
    dsf_root = root / "dsf"
    spec = importlib.util.spec_from_file_location("_factor_residual_dsf", dsf_root / "member_script.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(dsf_root))
    try:
        spec.loader.exec_module(module)
        module.DATA_DIR = DATA_DIR
        module.MODEL_DIR = str(dsf_root / "model")
        module.OUT_PATH = str(output.parent / "_dsf.csv")
        module.main()
    finally:
        sys.path.pop(0)

    raw = pd.read_csv(output.parent / "_dsf.csv")
    dsf = pd.Series(raw[TARGET].to_numpy(float), index=raw[ID].astype(str)).reindex(test[ID].astype(str)).to_numpy()
    model_dir = Path(MODEL_DIR)
    base_features = make_features(test)
    factor = _load_predict(
        model_dir / "factor_tabm.pt", joblib.load(model_dir / "preprocessor.joblib"), base_features
    )
    router = joblib.load(model_dir / "router.joblib")
    route_probability = router.predict_proba(_router_features(dsf, factor))[:, 1]
    command_artifacts = json.loads((model_dir / "trackman_command_artifacts.json").read_text())
    track_features = pd.concat(
        [base_features.reset_index(drop=True), enrich_command(test, command_artifacts).reset_index(drop=True)], axis=1
    )
    track = _load_predict(
        model_dir / "track_factor.pt", joblib.load(model_dir / "track_preprocessor.joblib"), track_features
    )
    metadata = json.loads((model_dir / "metadata.json").read_text())
    correction = float(metadata["router_alpha"]) * (factor - dsf)
    routed = route_probability >= float(metadata["router_threshold"])
    track_delta = track - dsf
    unsafe = (
        routed
        & (correction * track_delta < 0.0)
        & (np.abs(track_delta) >= float(metadata["track_strength_cutoff"]))
    )
    baseline = dsf + routed * correction
    baseline[unsafe] = dsf[unsafe]
    residual_features = _residual_features(dsf, factor, track, route_probability, routed, unsafe)
    residual = joblib.load(model_dir / "residual.joblib").predict(residual_features)
    prediction = baseline + float(metadata["residual_alpha"]) * np.clip(
        residual, -float(metadata["residual_cap"]), float(metadata["residual_cap"])
    )
    aligned = pd.Series(np.clip(prediction, 0.0, 1.0), index=test[ID].astype(str)).reindex(
        sample[ID].astype(str)
    ).to_numpy()
    if not np.isfinite(aligned).all():
        raise RuntimeError("row alignment failed")
    sample[TARGET] = aligned
    sample.to_csv(output, index=False)
    print(
        f"DSF/FactorTabM residual rows={len(sample)} routed={int(routed.sum())} blocked={int(unsafe.sum())}"
    )


if __name__ == "__main__":
    main()
