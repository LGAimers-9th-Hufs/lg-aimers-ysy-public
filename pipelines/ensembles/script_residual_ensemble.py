#!/usr/bin/env python3
"""전체기간/최근3시즌 CatBoost + embedding MLP OOF 최적 앙상블."""
from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DATA=Path("./data"); MODEL=Path("./model"); OUT=Path("./output/submission.csv")
ID,TARGET="row_id","control_success"
sys.path.insert(0,str((MODEL/"mlp").resolve()))
sys.path.insert(0,str(Path(".").resolve()))
import helper_catboost as cb
import helper_mlp as mlp

WEIGHTS=np.array([0.3877481476972833,0.27356474081163634,0.3386871114910804],dtype=np.float64)
CAL={"a":1.1063575942105441,"b":-0.029087104807525528}

def calibrate(p):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
 z=np.clip(CAL["a"]*np.log(p/(1-p))+CAL["b"],-30,30)
 return 1/(1+np.exp(-z))

def load_bundle():
 cms,ca=cb.load_bundle(MODEL/"catboost")
 mlp.MODEL=MODEL/"mlp"
 mm,pre,ma=mlp.load_bundle()
 return cms,ca,mm,pre,ma

def predict_frame(df,bundle):
 cms,ca,mm,pre,ma=bundle
 # CatBoost helper의 기존 내부 blend를 우회해 두 모델을 개별 예측한다.
 x=cb.build_features(df)
 if x.columns.tolist()!=ca["feature_cols"]: raise ValueError("CatBoost 피처 스키마 불일치")
 from catboost import Pool
 pool=Pool(x,cat_features=[x.columns.get_loc(c) for c in ca["cat_cols"]])
 pf=cms[0].predict_proba(pool,thread_count=6)[:,1]
 pr=cms[1].predict_proba(pool,thread_count=6)[:,1]
 pm=mlp.predict_frame(df,mm,pre,ma)
 return calibrate(WEIGHTS[0]*pf+WEIGHTS[1]*pr+WEIGHTS[2]*pm)

def main():
 torch.set_num_threads(6); bundle=load_bundle()
 test=pd.read_csv(DATA/"test.csv",encoding="utf-8-sig"); sub=pd.read_csv(DATA/"sample_submission.csv",encoding="utf-8-sig")
 if test[ID].isna().any() or not test[ID].is_unique: raise ValueError("test row_id 오류")
 if sub[ID].isna().any() or not sub[ID].is_unique: raise ValueError("submission row_id 오류")
 if len(test)!=len(sub) or set(test[ID].astype(str))!=set(sub[ID].astype(str)): raise ValueError("ID 불일치")
 p=predict_frame(test,bundle)
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("예측값 오류")
 sub[TARGET]=sub[ID].astype(str).map(dict(zip(test[ID].astype(str),p)))
 if sub[TARGET].isna().any(): raise ValueError("누락 예측")
 os.makedirs(OUT.parent,exist_ok=True); sub.to_csv(OUT,index=False,encoding="utf-8")
 print(f"saved {OUT}: rows={len(sub)}, mean={sub[TARGET].mean():.6f}")
if __name__=="__main__": main()
