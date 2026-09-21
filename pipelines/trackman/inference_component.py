#!/usr/bin/env python3
"""TrackMan 물리 CatBoost component의 행 독립 추론 helper."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool

MODEL=Path("./model/trackman"); sys.path.insert(0,str(MODEL.resolve()))
try:
 from shared.catboost_features import build_features
 from pipelines.trackman.features import enrich_trackman,load_artifacts
except ImportError:
 from catboost_features import build_features
 from trackman_features import enrich_trackman,load_artifacts

DROP_DIVERSITY={"pitcher_id","batter_id","team_matchup"}
def make_features(df,tm_art):
 x=build_features(df); x=x.drop(columns=[c for c in DROP_DIVERSITY if c in x]); tm=enrich_trackman(df,tm_art)
 for c in tm: x[c]=tm[c]
 p=pd.to_numeric(df["asof_pitcher_success_rate"],errors="coerce").fillna(.5)
 x["tm_skill_x_speed"]=(p*x["tm_exp_rel_speed"]).astype(np.float32)
 x["tm_skill_x_movement"]=(p*x["tm_movement_separation"]).astype(np.float32)
 x["tm_spin_x_breaking"]=(x["tm_exp_spin_rate"]*pd.to_numeric(df["asof_pitcher_breaking_rate"],errors="coerce").fillna(0)).astype(np.float32)
 return x

def load_bundle():
 with open(MODEL/"artifacts.json",encoding="utf-8") as f: meta=json.load(f)
 tm=load_artifacts(MODEL/meta["trackman_artifact"]); model=CatBoostClassifier(); model.load_model(MODEL/meta["model_file"]); return model,tm,meta

def predict_frame(df,bundle):
 model,tm,meta=bundle; x=make_features(df,tm)
 if x.columns.tolist()!=meta["feature_cols"]: raise ValueError("TrackMan 피처 스키마 불일치")
 pool=Pool(x,cat_features=[x.columns.get_loc(c) for c in meta["cat_cols"]])
 return model.predict_proba(pool,thread_count=6)[:,1]
