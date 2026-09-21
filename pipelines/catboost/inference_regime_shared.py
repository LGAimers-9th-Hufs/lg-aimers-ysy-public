#!/usr/bin/env python3
"""Regime shared-trunk CatBoost 행 독립 추론."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
DATA=Path("./data"); MODEL=Path("./model"); OUT=Path("./output/submission.csv"); sys.path.insert(0,str(MODEL.resolve()))
try:
 from shared.regime_features import build_features
except ImportError:
 from regime_features import build_features
ID,TARGET="row_id","control_success"
def load_bundle():
 with open(MODEL/"artifacts.json",encoding="utf-8") as f: a=json.load(f)
 m=CatBoostClassifier(); m.load_model(MODEL/"regime_shared.cbm"); return m,a
def predict_frame(df,m,a):
 x=build_features(df)
 if x.columns.tolist()!=a["feature_cols"]: raise ValueError("피처 스키마 불일치")
 ci=[x.columns.get_loc(c) for c in a["cat_cols"]]; p=m.predict_proba(Pool(x,cat_features=ci),thread_count=int(a.get("predict_threads",6)))[:,1]; c=a["calibration"]
 p=np.clip(p,1e-6,1-1e-6); return 1/(1+np.exp(-np.clip(c["a"]*np.log(p/(1-p))+c["b"],-30,30)))
def main():
 m,a=load_bundle(); test=pd.read_csv(DATA/"test.csv",encoding="utf-8-sig"); sub=pd.read_csv(DATA/"sample_submission.csv",encoding="utf-8-sig")
 if test[ID].isna().any() or not test[ID].is_unique or sub[ID].isna().any() or not sub[ID].is_unique: raise ValueError("row_id 오류")
 if len(test)!=len(sub) or set(test[ID].astype(str))!=set(sub[ID].astype(str)): raise ValueError("ID 불일치")
 p=predict_frame(test,m,a)
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("예측 오류")
 sub[TARGET]=sub[ID].astype(str).map(dict(zip(test[ID].astype(str),p))); os.makedirs(OUT.parent,exist_ok=True); sub.to_csv(OUT,index=False); print(f"saved {OUT}: rows={len(sub)}, mean={sub[TARGET].mean():.6f}")
if __name__=="__main__": main()
