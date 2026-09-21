#!/usr/bin/env python3
"""행의 game_type만으로 라우팅하는 CatBoost experts strict 평가."""
from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from scipy.optimize import minimize
from shared.catboost_features import ALL_CAT_COLS,ID_COL,TARGET_COL,build_features

def bss(y,p):
 y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64); bs=float(np.mean((p-y)**2)); ref=float(y.mean()*(1-y.mean()))
 return max(0.,100000*(1-bs/ref)),bs
def fit_cal(y,p):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6); y=np.asarray(y,dtype=np.float64); z=np.log(p/(1-p))
 def obj(ab):
  q=1/(1+np.exp(-np.clip(ab[0]*z+ab[1],-30,30))); return np.mean((q-y)**2)+5e-5*((ab[0]-1)**2+ab[1]**2)
 r=minimize(obj,[1.,0.],method="Nelder-Mead",options={"maxiter":1000,"fatol":1e-12}); return {"a":float(r.x[0]),"b":float(r.x[1])}
def calibrate(p,c):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6); return 1/(1+np.exp(-np.clip(c["a"]*np.log(p/(1-p))+c["b"],-30,30)))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--out",default="./game_type_experts.json"); a=ap.parse_args(); st=time.time()
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=build_features(df); y=df[TARGET_COL].to_numpy(dtype=np.int8); cats=[c for c in ALL_CAT_COLS if c in x]; ci=[x.columns.get_loc(c) for c in cats]
 va=df["season"].to_numpy()==2024; h=pd.util.hash_pandas_object(df.loc[va,ID_COL].astype(str),index=False).to_numpy(); stop=(h%4)<2; calfit=(h%4)==2; ev=(h%4)==3
 val_idx=df.index[va]; pred=pd.Series(index=val_idx,dtype=np.float64); details={}; calibrations={}
 for j,g in enumerate(["F","R"]):
  tr=(df["season"].to_numpy()<2024)&(df["game_type"].astype(str).to_numpy()==g); vg=df.loc[val_idx,"game_type"].astype(str).to_numpy()==g
  tr_year=df.loc[tr,"season"].to_numpy(); w=(1+.18*(tr_year-2019)).astype(np.float32); depth=7 if g=="F" else 8
  m=CatBoostClassifier(iterations=1400,depth=depth,learning_rate=.04,loss_function="Logloss",eval_metric="BrierScore",random_seed=9126+j*101,l2_leaf_reg=9.,random_strength=.55,bootstrap_type="Bernoulli",subsample=.87,rsm=.88,boosting_type="Ordered",one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,allow_writing_files=False)
  local_stop=stop[vg]; xv=x.loc[val_idx[vg]]; yv=y[va][vg]
  print(f"[{g}] train={tr.sum():,} val={vg.sum():,} stop={local_stop.sum():,}",flush=True)
  m.fit(Pool(x.loc[tr],y[tr],cat_features=ci,weight=w),eval_set=Pool(xv.loc[local_stop],yv[local_stop],cat_features=ci),early_stopping_rounds=140,use_best_model=True)
  pg=m.predict_proba(Pool(xv,cat_features=ci))[:,1]; pred.loc[val_idx[vg]]=pg
  local_cal=calfit[vg]; c=fit_cal(yv[local_cal],pg[local_cal]); calibrations[g]=c
  details[g]={"best_iteration":int(m.get_best_iteration()+1),"n_train":int(tr.sum()),"n_val":int(vg.sum())}
 raw=pred.to_numpy(); outp=raw.copy(); val_groups=df.loc[val_idx,"game_type"].astype(str).to_numpy()
 for g in ["F","R"]:
  mask=val_groups==g; outp[mask]=calibrate(raw[mask],calibrations[g])
 score_raw=bss(y[va][ev],raw[ev]); score=bss(y[va][ev],outp[ev])
 out={"experts":details,"calibrations":calibrations,"raw_bss":score_raw[0],"raw_brier":score_raw[1],"cal_bss":score[0],"cal_brier":score[1],"elapsed_seconds":time.time()-st,"rules":"official train only; per-row game_type routing; no test aggregates"}
 with open(a.out,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
