#!/usr/bin/env python3
"""V3 Brier 보조 모델의 행 독립 추론 helper."""
from __future__ import annotations
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
from helper_v3 import build_features
MODEL=Path("./model/brier")
def load_bundle():
 with open(MODEL/"artifacts.json",encoding="utf-8") as f: a=json.load(f)
 return lgb.Booster(model_file=str(MODEL/a["model_file"])),a
def predict_frame(df,b):
 m,a=b; x=build_features(df,a)
 if m.feature_name()!=a["feature_cols"]: raise ValueError("Brier 모델/artifact 불일치")
 p=np.clip(m.predict(x,num_threads=int(a.get("predict_threads",6))),1e-6,1-1e-6); c=a["calibration"]; z=np.clip(c["a"]*np.log(p/(1-p))+c["b"],-30,30); return 1/(1+np.exp(-z))
