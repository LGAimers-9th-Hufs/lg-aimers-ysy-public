#!/usr/bin/env python3
"""Evaluate a frozen TrackMan agreement gate around a FactorTabM router.

The TrackMan model is used only as a row-local safety signal.  All thresholds
are selected on temporal OOF select rows and are then frozen before the
untouched evaluate rows are scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_factor_router import brier, bss, router_features


def load(directory: str, year: int) -> dict[str, np.ndarray]:
    with np.load(Path(directory) / f"factor_{year}.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def fit_router(oof: dict[str, np.ndarray]) -> lgb.LGBMClassifier:
    better = ((oof["y"] - oof["factor"]) ** 2 < (oof["y"] - oof["dsf"]) ** 2).astype(int)
    model = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=1500,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=20.0,
        verbosity=-1,
        random_state=44123,
        n_jobs=6,
    )
    model.fit(router_features(oof)[oof["stop"]], better[oof["stop"]])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor_oof", default="./factor_tabm_oof_seed1")
    parser.add_argument("--track_oof", default="./factor_tabm_trackman_command_oof")
    parser.add_argument("--output", default="./trackman_safety_gate.json")
    args = parser.parse_args()

    factor = {year: load(args.factor_oof, year) for year in (2023, 2024)}
    track = {year: load(args.track_oof, year) for year in (2023, 2024)}
    for year in (2023, 2024):
        if not np.array_equal(factor[year]["row_id"], track[year]["row_id"]):
            raise RuntimeError(f"OOF alignment mismatch for {year}")

    router = fit_router(factor[2023])
    prepared: dict[int, dict[str, np.ndarray]] = {}
    select_strength = []
    for year in (2023, 2024):
        d = factor[year]
        route_probability = router.predict_proba(router_features(d))[:, 1]
        correction = 0.30 * (d["factor"] - d["dsf"])
        track_delta = track[year]["factor"] - d["dsf"]
        strength = np.abs(track_delta)
        agreement = correction * track_delta >= 0.0
        prepared[year] = {
            "base": d["dsf"] + (route_probability >= 0.50) * correction,
            "correction": (route_probability >= 0.50) * correction,
            "agreement": agreement,
            "strength": strength,
        }
        select_strength.append(strength[d["select"]])

    pooled_strength = np.concatenate(select_strength)
    cutoffs = sorted(set(float(np.quantile(pooled_strength, q)) for q in (0.0, 0.25, 0.50, 0.75)))
    candidates = []
    for cutoff in cutoffs:
        for disagreement_scale in (0.0, 0.25, 0.50, 0.75, 1.0):
            season_losses = []
            for year in (2023, 2024):
                d, p = factor[year], prepared[year]
                unsafe = (~p["agreement"]) & (p["strength"] >= cutoff)
                prediction = p["base"] - unsafe * (1.0 - disagreement_scale) * p["correction"]
                mask = d["select"]
                reference = d["y"][mask].mean() * (1.0 - d["y"][mask].mean())
                season_losses.append(brier(d["y"][mask], prediction[mask]) / reference)
            candidates.append((float(np.mean(season_losses)), cutoff, disagreement_scale))

    _, cutoff, disagreement_scale = min(candidates)
    result = {
        "factor_oof": args.factor_oof,
        "track_oof": args.track_oof,
        "router_threshold": 0.50,
        "router_alpha": 0.30,
        "track_strength_cutoff": cutoff,
        "disagreement_scale": disagreement_scale,
        "years": {},
    }
    for year in (2023, 2024):
        d, p = factor[year], prepared[year]
        unsafe = (~p["agreement"]) & (p["strength"] >= cutoff)
        prediction = p["base"] - unsafe * (1.0 - disagreement_scale) * p["correction"]
        mask = d["evaluate"]
        dsf_score = bss(d["y"][mask], d["dsf"][mask])
        base_score = bss(d["y"][mask], p["base"][mask])
        gated_score = bss(d["y"][mask], prediction[mask])
        result["years"][str(year)] = {
            "dsf_bss": dsf_score,
            "seed1_router_delta_bss": base_score - dsf_score,
            "safety_gate_delta_bss": gated_score - dsf_score,
            "safety_gate_vs_router_bss": gated_score - base_score,
            "unsafe_fraction": float(np.mean(unsafe[mask])),
        }

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
