#!/usr/bin/env python3
"""Sparse hierarchical logistic GAM with train-frozen row-local transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder


NUMERIC = [
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate", "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def _integer(frame, column, default=-1):
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(np.int64)


def categorical_matrix(frame: pd.DataFrame) -> np.ndarray:
    pitcher, batter = _integer(frame, "pitcher_id"), _integer(frame, "batter_id")
    pitcher_team, batter_team = _integer(frame, "pitcher_team_id"), _integer(frame, "batter_team_id")
    pitcher_hand, batter_hand = _integer(frame, "pitcher_hand"), _integer(frame, "batter_hand")
    balls, strikes = _integer(frame, "balls_before", 0), _integer(frame, "strikes_before", 0)
    count = balls * 3 + strikes
    game_type = frame["game_type"].astype(str).map({"F": 0, "R": 1}).fillna(-1).to_numpy(np.int64)
    base_state = frame["base_state"].astype(str).map({
        "___": 0, "1__": 1, "_2_": 2, "__3": 3,
        "12_": 4, "1_3": 5, "_23": 6, "123": 7,
    }).fillna(-1).to_numpy(np.int64)
    pn = np.digitize(pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0), [25, 100, 500, 1500])
    bn = np.digitize(pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0), [25, 100, 500, 1500])
    pr = np.digitize(pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").fillna(.5), [.40, .45, .48, .50, .52, .55, .60])
    br = np.digitize(pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce").fillna(.5), [.40, .45, .48, .50, .52, .55, .60])
    return np.column_stack([
        pitcher, batter, pitcher_team, batter_team, count, game_type, base_state,
        pitcher * 16 + count, batter * 8 + pitcher_hand,
        pitcher * 8 + batter_hand, (pitcher_team * 64 + batter_team) * 16 + count,
        pn, bn, pr, br,
    ])


class FrozenTransform:
    def fit(self, frame: pd.DataFrame):
        cats = categorical_matrix(frame)
        self.encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, dtype=np.float32)
        self.encoder.fit(cats)
        numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce")
        self.median = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.median)
        self.mean = filled.mean(); self.std = filled.std().replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame):
        cat = self.encoder.transform(categorical_matrix(frame))
        numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(self.median)
        numeric = ((numeric - self.mean) / self.std).clip(-8, 8).to_numpy(np.float32)
        # Explicit smooth low-order terms make the linear model an additive model.
        smooth = np.column_stack([numeric, numeric * numeric, np.tanh(numeric)]).astype(np.float32)
        return sparse.hstack([cat, sparse.csr_matrix(smooth)], format="csr", dtype=np.float32)
