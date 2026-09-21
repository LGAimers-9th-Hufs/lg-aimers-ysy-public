#!/usr/bin/env python3
"""원본 V3 90% + 동일 피처 직접 Brier regressor 10%."""
from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
DATA=Path("./data"); MODEL=Path("./model"); OUT=Path("./output/submission.csv"); ID,TARGET="row_id","control_success"
sys.path.insert(0,str(Path(".").resolve())); import helper_v3 as v3; import helper_brier as br
def load_bundle(): br.MODEL=MODEL/"brier"; return (*v3.load_bundle(MODEL/"v3"),br.load_bundle())
def predict_frame(df,b): vm,va,bb=b; return .9*v3.predict_frame(df,vm,va)+.1*br.predict_frame(df,bb)
def main():
 b=load_bundle(); test=pd.read_csv(DATA/"test.csv",encoding="utf-8-sig"); sub=pd.read_csv(DATA/"sample_submission.csv",encoding="utf-8-sig")
 if test[ID].isna().any() or not test[ID].is_unique or sub[ID].isna().any() or not sub[ID].is_unique: raise ValueError("row_id 오류")
 if len(test)!=len(sub) or set(test[ID].astype(str))!=set(sub[ID].astype(str)): raise ValueError("ID 불일치")
 p=predict_frame(test,b)
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("예측값 오류")
 sub[TARGET]=sub[ID].astype(str).map(dict(zip(test[ID].astype(str),p)))
 if sub[TARGET].isna().any(): raise ValueError("누락 예측")
 os.makedirs(OUT.parent,exist_ok=True); sub.to_csv(OUT,index=False,encoding="utf-8"); print(f"saved {OUT}: rows={len(sub)}, mean={sub[TARGET].mean():.6f}")
if __name__=="__main__": main()
