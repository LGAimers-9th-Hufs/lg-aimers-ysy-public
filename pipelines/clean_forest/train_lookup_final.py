#!/usr/bin/env python3
"""Freeze 2024 official-train residual lookups for 2025 inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.clean_forest.evaluate_residual_lookup import GROUPS, align, key_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--clean_oof", default="/private/tmp/clean_moe_2024_oof.npz")
    parser.add_argument("--regime_oof", default="/private/tmp/regime_stack_2024_oof.npz")
    parser.add_argument("--lookup_selection", default="/private/tmp/clean_lookup_eval/lookup_model.json")
    parser.add_argument("--output", default="/private/tmp/clean_lookup_final/lookup.json")
    args = parser.parse_args()
    columns = sorted({column for group in GROUPS.values() for column in group})
    frame = pd.read_csv(
        args.train_path, usecols=list(dict.fromkeys(["row_id", "season", "control_success", *columns])),
        encoding="utf-8-sig", low_memory=False,
    )
    frame = frame.loc[frame["season"].eq(2024)].copy()
    ids = frame["row_id"].astype(str).to_numpy()
    clean = align(Path(args.clean_oof), ids)
    regime = align(Path(args.regime_oof), ids)
    base = np.clip(0.70 * clean["prediction"] + 0.30 * regime["prediction"] - 0.002, 0.0, 1.0)
    residual = frame["control_success"].to_numpy(float) - base
    selected = json.loads(Path(args.lookup_selection).read_text(encoding="utf-8"))
    statistics = {}
    for name, columns_for_group in GROUPS.items():
        grouped = pd.DataFrame({"key": key_frame(frame, columns_for_group), "residual": residual}).groupby("key", sort=False)["residual"].agg(["sum", "count"])
        statistics[name] = {
            "columns": columns_for_group,
            "sum": {str(key): float(value) for key, value in grouped["sum"].items()},
            "count": {str(key): int(value) for key, value in grouped["count"].items()},
        }
    artifact = {
        "version": "clean-lookup-v1",
        "training_season": 2024,
        "kappas": selected["kappas"],
        "feature_names": selected["feature_names"],
        "scale": selected["scale"],
        "coef": selected["coef"],
        "strength": selected["strength"],
        "cap": 0.03,
        "statistics": statistics,
        "base": "clean_regime; weights 0.70 clean_moe + 0.30 regime - 0.002",
        "selection": "ridge alpha/strength selected by 2024 select inner hash split; 2024 evaluate untouched",
        "provenance": "official train.csv labels and OOF predictions only; no leaderboard, external data, or test-row aggregation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"saved {output} bytes={output.stat().st_size:,} groups={len(statistics)}")


if __name__ == "__main__":
    main()
