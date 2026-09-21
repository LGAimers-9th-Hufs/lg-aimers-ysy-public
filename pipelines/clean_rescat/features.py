"""Row-local features for the clean residual CatBoost."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "control_success"
CATEGORICAL = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = frame.drop(columns=[c for c in ("row_id", "season", TARGET) if c in frame]).copy()
    x["count_code"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["count_hand"] = x["count_code"] + "|" + x["pitcher_hand"].astype(str) + "|" + x["batter_hand"].astype(str)
    x["count_base"] = x["count_code"] + "|" + x["base_state"].astype(str)
    x["regime_count"] = x["game_type"].astype(str) + "|" + x["count_code"]
    x["hand_match"] = (x["pitcher_hand"].astype(str) == x["batter_hand"].astype(str)).astype(str)
    x["p_logn"] = np.log1p(pd.to_numeric(x["asof_pitcher_n"], errors="coerce").fillna(0.0))
    x["b_logn"] = np.log1p(pd.to_numeric(x["asof_batter_n"], errors="coerce").fillna(0.0))
    x["p_form_shock"] = (
        pd.to_numeric(x["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(x["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    )
    cats = [*CATEGORICAL, "count_code", "count_hand", "count_base", "regime_count", "hand_match"]
    for column in cats:
        x[column] = x[column].astype("string").fillna("<NA>").astype(str)
    return x, cats
