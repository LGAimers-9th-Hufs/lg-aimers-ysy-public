"""CatBoost 공용 피처: 같은 행의 값만 사용하며 집계는 수행하지 않는다."""
from __future__ import annotations
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
]
DROP_COLS = [ID_COL, TARGET_COL]

def _num(df, col, default=np.nan):
    if col not in df:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").astype(np.float32)

def build_features(df):
    """test 행 간 참조 없이 행별 피처만 만든다."""
    required = [c for c in CAT_COLS if c not in df]
    if required:
        raise ValueError(f"입력 컬럼 누락: {required}")
    raw = df.drop(columns=[c for c in DROP_COLS if c in df], errors="ignore")
    x = pd.DataFrame(index=df.index)
    for c in raw.columns:
        if c in CAT_COLS:
            x[c] = raw[c].fillna("__MISSING__").astype(str)
        else:
            x[c] = pd.to_numeric(raw[c], errors="coerce").astype(np.float32)

    balls = _num(df,"balls_before",0).fillna(0)
    strikes = _num(df,"strikes_before",0).fillna(0)
    inning = _num(df,"inning",1).fillna(1)
    runners = _num(df,"num_runners_on",0).fillna(0)
    li = _num(df,"li",1).fillna(1)
    score = _num(df,"score_diff_pitcher_team",0).fillna(0)
    p_n = _num(df,"asof_pitcher_n",0).fillna(0).clip(lower=0)
    b_n = _num(df,"asof_batter_n",0).fillna(0).clip(lower=0)
    p = _num(df,"asof_pitcher_success_rate").fillna(.52)
    b = _num(df,"asof_batter_success_rate").fillna(.52)

    x["count_state"] = (balls.astype(np.int16)*4+strikes.astype(np.int16)).astype(str)
    x["runner_state"] = ((_num(df,"runner_on_1b",0).fillna(0)>0).astype(np.int8)
        + 2*(_num(df,"runner_on_2b",0).fillna(0)>0).astype(np.int8)
        + 4*(_num(df,"runner_on_3b",0).fillna(0)>0).astype(np.int8)).astype(str)
    x["team_matchup"] = df["pitcher_team_id"].astype(str)+"_"+df["batter_team_id"].astype(str)
    x["hand_matchup"] = df["pitcher_hand"].astype(str)+"_"+df["batter_hand"].astype(str)
    x["inning_bucket"] = np.minimum(inning,10).astype(np.int16).astype(str)

    x["p_logn"] = np.log1p(p_n).astype(np.float32)
    x["b_logn"] = np.log1p(b_n).astype(np.float32)
    x["p_minus_b"] = (p-b).astype(np.float32)
    x["p_x_b"] = (p*b).astype(np.float32)
    x["pressure"] = (li*(1+(balls>=3).astype(np.float32))*(1+runners)).astype(np.float32)
    x["count_pressure"] = ((balls-strikes)*li).astype(np.float32)
    x["score_abs"] = score.abs().astype(np.float32)
    x["late_close"] = ((inning>=7)&(score.abs()<=2)).astype(np.int8)

    p1 = _num(df,"asof_pitcher_prev1_game_success_rate").fillna(p)
    p3 = _num(df,"asof_pitcher_prev3_game_success_rate").fillna(p)
    p5 = _num(df,"asof_pitcher_prev5_game_success_rate").fillna(p)
    m1 = _num(df,"asof_pitcher_prev1_game_middle_rate").fillna(_num(df,"asof_pitcher_middle_rate").fillna(.3))
    m5 = _num(df,"asof_pitcher_prev5_game_middle_rate").fillna(m1)
    x["success_trend_1_5"] = (p1-p5).astype(np.float32)
    x["success_trend_3_5"] = (p3-p5).astype(np.float32)
    x["recent_vs_career"] = (p3-p).astype(np.float32)
    x["middle_trend_1_5"] = (m1-m5).astype(np.float32)
    x["recent_x_li"] = (p1*li).astype(np.float32)

    fb = _num(df,"asof_pitcher_fastball_rate").fillna(.5).clip(lower=0)
    br = _num(df,"asof_pitcher_breaking_rate").fillna(.25).clip(lower=0)
    off = _num(df,"asof_pitcher_offspeed_rate").fillna(.25).clip(lower=0)
    total=(fb+br+off).replace(0,1); fb,br,off=fb/total,br/total,off/total
    x["mix_entropy"] = (-(fb*np.log(fb+1e-7)+br*np.log(br+1e-7)+off*np.log(off+1e-7))).astype(np.float32)
    return x

EXTRA_CAT_COLS = ["count_state","runner_state","team_matchup","hand_matchup","inning_bucket"]
ALL_CAT_COLS = CAT_COLS + EXTRA_CAT_COLS
