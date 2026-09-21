#!/usr/bin/env python3
"""CatBoost 2 + embedding MLP + 공식 TrackMan 물리 모델 OOF 앙상블."""
from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DATA=Path("./data"); MODEL=Path("./model"); OUT=Path("./output/submission.csv"); ID,TARGET="row_id","control_success"
sys.path.insert(0,str((MODEL/"mlp").resolve())); sys.path.insert(0,str((MODEL/"trackman").resolve())); sys.path.insert(0,str(Path(".").resolve()))
import helper_catboost as cb
import helper_mlp as mlp
import helper_trackman as tm

WEIGHTS=np.array([0.216973156432481,0.18043556763228497,0.27787390284490743,0.3247173730903267],dtype=np.float64)
CAL={"a":1.1289776528248188,"b":-0.028175067581563663}
def calibrate(p):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6); z=np.clip(CAL["a"]*np.log(p/(1-p))+CAL["b"],-30,30); return 1/(1+np.exp(-z))
def load_bundle():
 cms,ca=cb.load_bundle(MODEL/"catboost"); mlp.MODEL=MODEL/"mlp"; mm,pre,ma=mlp.load_bundle(); tm.MODEL=MODEL/"trackman"; tb=tm.load_bundle(); return cms,ca,mm,pre,ma,tb
def predict_frame(df,b):
 cms,ca,mm,pre,ma,tb=b; x=cb.build_features(df)
 if x.columns.tolist()!=ca["feature_cols"]: raise ValueError("CatBoost 피처 스키마 불일치")
 from catboost import Pool
 pool=Pool(x,cat_features=[x.columns.get_loc(c) for c in ca["cat_cols"]]); pf=cms[0].predict_proba(pool,thread_count=6)[:,1]; pr=cms[1].predict_proba(pool,thread_count=6)[:,1]
 pm=mlp.predict_frame(df,mm,pre,ma); pt=tm.predict_frame(df,tb); return calibrate(WEIGHTS[0]*pf+WEIGHTS[1]*pr+WEIGHTS[2]*pm+WEIGHTS[3]*pt)
def main():
 torch.set_num_threads(6); b=load_bundle(); test=pd.read_csv(DATA/"test.csv",encoding="utf-8-sig"); sub=pd.read_csv(DATA/"sample_submission.csv",encoding="utf-8-sig")
 if test[ID].isna().any() or not test[ID].is_unique or sub[ID].isna().any() or not sub[ID].is_unique: raise ValueError("row_id 오류")
 if len(test)!=len(sub) or set(test[ID].astype(str))!=set(sub[ID].astype(str)): raise ValueError("ID 불일치")
 p=predict_frame(test,b)
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("예측값 오류")
 sub[TARGET]=sub[ID].astype(str).map(dict(zip(test[ID].astype(str),p)))
 if sub[TARGET].isna().any(): raise ValueError("누락 예측")
 os.makedirs(OUT.parent,exist_ok=True); sub.to_csv(OUT,index=False,encoding="utf-8"); print(f"saved {OUT}: rows={len(sub)}, mean={sub[TARGET].mean():.6f}")
if __name__=="__main__": main()
