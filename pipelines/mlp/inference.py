#!/usr/bin/env python3
"""PyTorch embedding MLP 행 독립 추론."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DATA=Path("./data"); MODEL=Path("./model"); OUT=Path("./output/submission.csv"); sys.path.insert(0,str(MODEL.resolve()))
try:
 from pipelines.mlp.train_walkforward import SharedTrunkMLP,enrich
except ImportError:  # 제출 ZIP에서는 helper가 model/에 동봉된다.
 from train_mlp_walkforward import SharedTrunkMLP,enrich

ID,TARGET="row_id","control_success"

class FrozenPreprocessor:
 def __init__(self,a):
  self.cat_cols=a["cat_cols"]; self.num_cols=a["num_cols"]; self.maps=a["maps"]
  self.median=pd.Series(a["median"]); self.mean=pd.Series(a["mean"]); self.std=pd.Series(a["std"])
 def transform(self,x):
  cats=np.column_stack([x[c].fillna("__MISSING__").astype(str).map(self.maps[c]).fillna(0).to_numpy(np.int64) for c in self.cat_cols])
  num=x[self.num_cols].apply(pd.to_numeric,errors="coerce").fillna(self.median); num=((num-self.mean)/self.std).clip(-8,8).to_numpy(np.float32)
  return cats,num

def load_bundle():
 with open(MODEL/"artifacts.json",encoding="utf-8") as f: art=json.load(f)
 with open(MODEL/"preprocess.json",encoding="utf-8") as f: prep=json.load(f)
 ck=torch.load(MODEL/"mlp.pt",map_location="cpu",weights_only=True); model=SharedTrunkMLP(ck["cardinalities"],ck["n_num"]); model.load_state_dict(ck["state_dict"]); model.eval(); return model,FrozenPreprocessor(prep),art

def predict_frame(df,model,pre,art):
 x=enrich(df)
 if x.columns.tolist()!=art["feature_cols"]: raise ValueError("피처 스키마 불일치")
 cats,num=pre.transform(x); out=[]
 with torch.no_grad():
  for s in range(0,len(df),16384):
   out.append(torch.sigmoid(model(torch.as_tensor(cats[s:s+16384]),torch.as_tensor(num[s:s+16384]))).numpy())
 return np.concatenate(out)

def main():
 torch.set_num_threads(6); model,pre,art=load_bundle(); test=pd.read_csv(DATA/"test.csv",encoding="utf-8-sig"); sub=pd.read_csv(DATA/"sample_submission.csv",encoding="utf-8-sig")
 if test[ID].isna().any() or not test[ID].is_unique: raise ValueError("test row_id 오류")
 if sub[ID].isna().any() or not sub[ID].is_unique: raise ValueError("submission row_id 오류")
 if len(test)!=len(sub) or set(test[ID].astype(str))!=set(sub[ID].astype(str)): raise ValueError("ID 불일치")
 p=predict_frame(test,model,pre,art)
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("예측값 오류")
 sub[TARGET]=sub[ID].astype(str).map(dict(zip(test[ID].astype(str),p)))
 if sub[TARGET].isna().any(): raise ValueError("누락 예측")
 os.makedirs(OUT.parent,exist_ok=True); sub.to_csv(OUT,index=False,encoding="utf-8"); print(f"saved {OUT}: rows={len(sub)}, mean={sub[TARGET].mean():.6f}")
if __name__=="__main__": main()
