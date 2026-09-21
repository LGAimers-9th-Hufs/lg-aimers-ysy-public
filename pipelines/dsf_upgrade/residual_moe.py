#!/usr/bin/env python3
"""Shared row-local features for the 2024 residual mixture-of-experts."""
from __future__ import annotations

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b",
    "runner_on_3b", "num_runners_on", "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def make_moe_features(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    factors: np.ndarray,
    track: np.ndarray,
    route_probability: np.ndarray,
    current_baseline: np.ndarray,
    current_residual: np.ndarray,
) -> np.ndarray:
    """Create only row-wise features; no statistics are fitted on ``frame``."""
    raw = np.column_stack([_numeric(frame, column) for column in RAW_COLUMNS])
    balls = np.nan_to_num(_numeric(frame, "balls_before"), nan=0.0)
    strikes = np.nan_to_num(_numeric(frame, "strikes_before"), nan=0.0)
    pitcher_n = np.maximum(np.nan_to_num(_numeric(frame, "asof_pitcher_n"), nan=0.0), 0.0)
    batter_n = np.maximum(np.nan_to_num(_numeric(frame, "asof_batter_n"), nan=0.0), 0.0)
    pitchmix_n = np.maximum(np.nan_to_num(_numeric(frame, "asof_pitcher_pitchmix_n"), nan=0.0), 0.0)
    pitcher_rate = _numeric(frame, "asof_pitcher_success_rate")
    batter_rate = _numeric(frame, "asof_batter_success_rate")
    prev1 = _numeric(frame, "asof_pitcher_prev1_game_success_rate")
    prev3 = _numeric(frame, "asof_pitcher_prev3_game_success_rate")
    prev5 = _numeric(frame, "asof_pitcher_prev5_game_success_rate")
    game_type = frame["game_type"].astype(str).map({"F": 0.0, "R": 1.0}).fillna(-1.0).to_numpy()
    top_bottom = frame["top_bottom"].astype(str).map({"T": 0.0, "B": 1.0}).fillna(-1.0).to_numpy()
    base_state = frame["base_state"].astype(str).map(
        {"___": 0.0, "1__": 1.0, "_2_": 2.0, "__3": 3.0, "12_": 4.0, "1_3": 5.0, "_23": 6.0, "123": 7.0}
    ).fillna(-1.0).to_numpy()
    derived = np.column_stack(
        [
            game_type,
            top_bottom,
            base_state,
            balls * 3.0 + strikes,
            (balls >= 3.0).astype(float),
            (strikes >= 2.0).astype(float),
            ((balls >= 3.0) & (strikes >= 2.0)).astype(float),
            np.log1p(pitcher_n),
            np.log1p(batter_n),
            np.log1p(pitchmix_n),
            pitcher_rate - batter_rate,
            prev1 - pitcher_rate,
            prev3 - pitcher_rate,
            prev5 - pitcher_rate,
            prev1 - prev5,
        ]
    )
    factor_mean = factors.mean(axis=1)
    factor_std = factors.std(axis=1)
    predictions = np.column_stack(
        [
            base_prediction,
            factors,
            track,
            current_baseline,
            current_residual,
            route_probability,
            factor_mean,
            factor_std,
            factors - base_prediction[:, None],
            track - base_prediction,
            factor_mean - base_prediction,
            np.abs(factors - base_prediction[:, None]),
            np.abs(track - base_prediction),
            np.abs(current_baseline - base_prediction),
            (factors[:, 1] - base_prediction) * (track - base_prediction),
        ]
    )
    return np.asarray(np.column_stack([predictions, raw, derived]), dtype=np.float32)
