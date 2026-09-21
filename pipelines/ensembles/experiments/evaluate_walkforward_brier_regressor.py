#!/usr/bin/env python3
"""Brier 직접 최적화 CatBoostRegressor walk-forward 평가."""
from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
from scipy.optimize import minimize
from shared.catboost_features import ALL_CAT_COLS,TARGET_COL,build_features

def bss(y,p):
 y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
 bs=float(np.mean((p-y)**2)); ref=float(y.mean()*(1-y.mean()))
 return max(0.,100000*(1-bs/ref)),bs

def fit_affine(y,p):
 y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
 def obj(ab): return np.mean((np.clip(ab[0]*p+ab[1],0,1)-y)**2)+2e-5*((ab[0]-1)**2+ab[1]**2)
 r=minimize(obj,[1.,0.],method="Nelder-Mead",options={"maxiter":1000,"fatol":1e-12})
 return {"a":float(r.x[0]),"b":float(r.x[1])}

def apply_affine(p,c): return np.clip(c["a"]*np.asarray(p)+c["b"],0,1)

def params(seed,it):
 return dict(iterations=it,depth=8,learning_rate=.04,loss_function="RMSE",eval_metric="RMSE",random_seed=seed,
  l2_leaf_reg=9.,random_strength=.5,bootstrap_type="Bernoulli",subsample=.86,rsm=.88,boosting_type="Ordered",
  one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,allow_writing_files=False)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--out",default="./walkforward_brier_regressor.json"); ap.add_argument("--iterations",type=int,default=240)
 a=ap.parse_args(); start=time.time(); df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=build_features(df); y=df[TARGET_COL].to_numpy(dtype=np.float32)
 cats=[c for c in ALL_CAT_COLS if c in x]; cat_idx=[x.columns.get_loc(c) for c in cats]; preds={}; result={}
 for year in [2022,2023,2024]:
  tr=df["season"].to_numpy()<year; va=df["season"].to_numpy()==year
  w=(1+.18*(df.loc[tr,"season"].to_numpy()-2019)).astype(np.float32)
  print(f"[{year}] Brier regressor train={tr.sum():,} val={va.sum():,}",flush=True)
  m=CatBoostRegressor(**params(5026+year,a.iterations)); m.fit(Pool(x.loc[tr],y[tr],cat_features=cat_idx,weight=w))
  raw=m.predict(Pool(x.loc[va],cat_features=cat_idx)); clipped=np.clip(raw,0,1); preds[year]=clipped
  score=bss(y[va],clipped); result[str(year)]={"raw_bss":score[0],"raw_brier":score[1],"unclipped_min":float(raw.min()),"unclipped_max":float(raw.max())}
  print(f"[{year}] BSS={score[0]:.3f} range={raw.min():.4f}..{raw.max():.4f}")
 fold=[]
 for year,cal_year in [(2023,2022),(2024,2023)]:
  yc=df.loc[df["season"]==cal_year,TARGET_COL].to_numpy(); ye=df.loc[df["season"]==year,TARGET_COL].to_numpy()
  cal=fit_affine(yc,preds[cal_year]); score=bss(ye,apply_affine(preds[year],cal)); result[str(year)]["prior_year_cal_bss"]=score[0]; result[str(year)]["prior_year_cal_brier"]=score[1]; fold.append(score[0])
 result["summary"]={"raw_mean_bss":float(np.mean([result[str(y)]["raw_bss"] for y in [2022,2023,2024]])),"raw_std_bss":float(np.std([result[str(y)]["raw_bss"] for y in [2022,2023,2024]])),"prior_year_cal_mean_bss_2023_2024":float(np.mean(fold)),"prior_year_cal_worst_bss":float(np.min(fold))}
 result["iterations"]=a.iterations; result["elapsed_seconds"]=time.time()-start; result["rules"]="official train only; test rows independent"
 with open(a.out,"w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
