#!/usr/bin/env python3
"""Fit the frozen clean regime stack through 2024 for 2025 inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.regime_stack.train import (
    TARGET, artifact, category_maps, feature_groups, fit_models,
    load_feature_module, time_safe_history,
)


FROZEN_ROUNDS = {"specialist": 95, "state": 300, "invariant": 230}
FROZEN_MIXTURE = {"specialist": 0.10, "state": 0.05, "invariant": 0.85, "shift": -0.014}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--output_dir", default="./regime_stack_final")
    args = parser.parse_args()
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    feature_module = load_feature_module()
    maps = category_maps(frame)
    groups = feature_groups(feature_module)
    x = time_safe_history(frame, maps, feature_module, groups["specialist"])
    models = fit_models(
        x, frame[TARGET].to_numpy(np.float32), frame["season"].to_numpy(int),
        groups, feature_module,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.booster_.save_model(str(output / f"{name}.txt"), num_iteration=FROZEN_ROUNDS[name])
    metadata = artifact(frame, maps)
    metadata.update({
        "feature_names": groups,
        "rounds": FROZEN_ROUNDS,
        "mixture": FROZEN_MIXTURE,
        "training_through": 2024,
        "training": "specialist=2024 only; state/invariant=2019-2024 with 0.50 annual decay",
        "selection": "rounds and mixture selected on 2024 select; 2024 evaluate untouched",
        "provenance": "official train.csv only; no DSF predictions/models, TrackMan, external data, or leaderboard fitting",
    })
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output.resolve()} rows={len(frame):,}")


if __name__ == "__main__":
    main()
