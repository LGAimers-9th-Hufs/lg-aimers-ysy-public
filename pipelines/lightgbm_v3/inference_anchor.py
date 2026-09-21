#!/usr/bin/env python3
"""실제 최고 V3 80% + 규정 준수 CatBoost/MLP 20% 앙상블."""
from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DATA=Path("./data"); MODEL=Path("./model"); OUT=Path("./output/submission.csv"); ID,TARGET="row_id","control_success"
sys.path.insert(0,str((MODEL/"residual"/"mlp").resolve())); sys.path.insert(0,str(Path(".").resolve()))
import helper_v3 as v3
import helper_residual as residual

V3_WEIGHT=.80
def load_bundle():
 vm,va=v3.load_bundle(MODEL/"v3"); residual.MODEL=MODEL/"residual"; rb=residual.load_bundle(); return vm,va,rb
def predict_frame(df,b):
 vm,va,rb=b; pv=v3.predict_frame(df,vm,va); pr=residual.predict_frame(df,rb)
 # 두 component는 각자의 train-only OOF calibration이 이미 적용되어 있다.
 return V3_WEIGHT*pv+(1-V3_WEIGHT)*pr
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
