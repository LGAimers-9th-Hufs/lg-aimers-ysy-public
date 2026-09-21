#!/usr/bin/env python3
"""Audit temporally stable DSF failure segments without using test-row statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROUTER_WEIGHTS = {"0-0": 0.19, "0-1": 0.10, "1-2": 0.10, "2-1": 0.40, "3-1": 0.14}


def normalized_gain(y, reference, candidate):
    y = np.asarray(y, float)
    denominator = y.mean() * (1.0 - y.mean())
    return float(100000.0 * np.mean((y - reference) ** 2 - (y - candidate) ** 2) / denominator)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--dsf_oof", required=True)
    ap.add_argument("--output", default="./dsf_segment_audit.json")
    args = ap.parse_args()

    columns = ["row_id", "season", "game_type", "balls_before", "strikes_before", "game_month"]
    train = pd.read_csv(args.train_path, usecols=columns, encoding="utf-8-sig")
    train["row_id"] = train["row_id"].astype(str)
    with np.load(args.dsf_oof, allow_pickle=False) as saved:
        oof = pd.DataFrame({key: saved[key] for key in (
            "row_id", "season", "y", "champion_reconstructed", "challenger", "blend_061"
        )})
    oof["row_id"] = oof["row_id"].astype(str)
    frame = oof.merge(train, on=["row_id", "season"], validate="one_to_one")
    frame["count_state"] = frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)

    weight = np.full(len(frame), 0.061, dtype=float)
    regular = frame["game_type"].eq("R")
    weight[regular] = frame.loc[regular, "count_state"].map(ROUTER_WEIGHTS).fillna(0.061)
    routed = (1.0 - weight) * frame["champion_reconstructed"] + weight * frame["challenger"]

    report = {"weights": ROUTER_WEIGHTS, "season_gain_bss": {}, "month_2024_gain_bss": {}}
    for year in (2022, 2023, 2024):
        mask = frame["season"].eq(year)
        report["season_gain_bss"][str(year)] = normalized_gain(
            frame.loc[mask, "y"], frame.loc[mask, "blend_061"], routed[mask]
        )
    block = frame.loc[frame["season"].eq(2024)].copy()
    block["routed"] = routed[frame["season"].eq(2024)].to_numpy()
    for month, group in block.groupby("game_month"):
        report["month_2024_gain_bss"][str(int(month))] = normalized_gain(
            group["y"], group["blend_061"], group["routed"]
        )
    report["accepted"] = bool(
        report["season_gain_bss"]["2024"] >= 5.0
        and all(value >= 0 for value in report["month_2024_gain_bss"].values())
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
