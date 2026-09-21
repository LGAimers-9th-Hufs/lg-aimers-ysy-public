"""행 단위 game_type/regime shared-trunk 피처."""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from shared.catboost_features import build_features as base_features, ALL_CAT_COLS
except ImportError:
    from catboost_features import build_features as base_features, ALL_CAT_COLS

REGIME_CAT_COLS = ALL_CAT_COLS + [
    "regime", "game_type_regime", "game_type_pitcher_team",
    "game_type_batter_team", "game_type_count", "regime_count",
]

def _num(df,col,default=0.):
    if col not in df: return pd.Series(default,index=df.index,dtype=np.float32)
    return pd.to_numeric(df[col],errors="coerce").fillna(default).astype(np.float32)

def build_features(df):
    x=base_features(df)
    season=_num(df,"season",2019); post=(season>=2023).astype(np.float32)
    regime=np.where(post>0,"post23","pre23"); gt=df["game_type"].fillna("__MISSING__").astype(str)
    pt=df["pitcher_team_id"].fillna("__MISSING__").astype(str); bt=df["batter_team_id"].fillna("__MISSING__").astype(str)
    x["regime"]=regime
    x["game_type_regime"]=gt+"_"+pd.Series(regime,index=df.index).astype(str)
    x["game_type_pitcher_team"]=gt+"_"+pt
    x["game_type_batter_team"]=gt+"_"+bt
    x["game_type_count"]=gt+"_"+x["count_state"].astype(str)
    x["regime_count"]=pd.Series(regime,index=df.index).astype(str)+"_"+x["count_state"].astype(str)
    f=(gt=="F").astype(np.float32); x["post23_flag"]=post; x["game_type_f_flag"]=f
    x["post23_x_game_type_f"]=(post*f).astype(np.float32)
    for source in ["asof_pitcher_success_rate","asof_pitcher_ball_rate","asof_pitcher_middle_rate","asof_pitcher_reverse_rate","asof_pitcher_prev3_game_success_rate","li"]:
        v=_num(df,source,0.)
        x[f"post23_x_{source}"]=(post*v).astype(np.float32)
        x[f"game_type_f_x_{source}"]=(f*v).astype(np.float32)
    return x
