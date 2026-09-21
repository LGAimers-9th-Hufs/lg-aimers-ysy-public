#!/usr/bin/env python3
"""Fixed row-aligned blend of the frozen DSF submission and V3."""
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


def _run(root: Path, name: str, output: Path) -> pd.DataFrame:
    member = root / name
    script = member / "member_script.py"
    spec = importlib.util.spec_from_file_location(f"_dsf_upgrade_{name}", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load member: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(member))
    try:
        spec.loader.exec_module(module)
        if name == "v3":
            # V3's load_bundle default is bound when the module is imported, so
            # pass the nested model directory explicitly instead of calling main.
            test = pd.read_csv(Path(DATA_DIR, "test.csv"), encoding="utf-8-sig")
            sample = pd.read_csv(Path(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
            models, artifact = module.load_bundle(member / "model")
            prediction = module.predict_frame(test, models, artifact)
            values = pd.Series(prediction, index=test[ID_COL].astype(str))
            sample[TARGET_COL] = sample[ID_COL].astype(str).map(values)
            if sample[TARGET_COL].isna().any():
                raise RuntimeError("V3 row alignment failed")
            sample.to_csv(output, index=False, encoding="utf-8")
        else:
            module.DATA_DIR = DATA_DIR
            module.MODEL_DIR = str(member / "model")
            module.OUT_PATH = str(output)
            module.main()
    finally:
        sys.path.pop(0)
    if not output.is_file():
        raise RuntimeError(f"member did not create output: {name}")
    return pd.read_csv(output, encoding="utf-8-sig")


def _align(frame: pd.DataFrame, ids: pd.Series, name: str) -> np.ndarray:
    if ID_COL not in frame or TARGET_COL not in frame:
        raise RuntimeError(f"invalid output columns: {name}")
    if frame[ID_COL].isna().any() or not frame[ID_COL].is_unique:
        raise RuntimeError(f"invalid row_id: {name}")
    values = pd.Series(
        pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(np.float64),
        index=frame[ID_COL].astype(str),
    ).reindex(ids.astype(str)).to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"row alignment failed: {name}")
    return values


def main() -> None:
    root = Path(__file__).resolve().parent
    metadata = json.loads(Path(MODEL_DIR, "metadata.json").read_text(encoding="utf-8"))
    sample = pd.read_csv(Path(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    output_dir = Path(OUT_PATH).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    dsf = _align(_run(root, "dsf", output_dir / "_dsf.csv"), sample[ID_COL], "dsf")
    v3 = _align(_run(root, "v3", output_dir / "_v3.csv"), sample[ID_COL], "v3")
    v3_weight = float(metadata["v3_weight"])
    prediction = np.clip((1.0 - v3_weight) * dsf + v3_weight * v3, 0.0, 1.0)
    result = sample.copy()
    result[TARGET_COL] = prediction
    result.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"DSF/V3 fixed blend rows={len(result)} v3_weight={v3_weight:.3f}")


if __name__ == "__main__":
    main()
