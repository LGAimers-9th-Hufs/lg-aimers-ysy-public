#!/usr/bin/env python3
"""game_type을 분리하지 않는 regime-interaction CatBoost shared trunk."""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from scipy.optimize import minimize
from shared.regime_features import build_features,REGIME_CAT_COLS
from shared.catboost_features import ID_COL,TARGET_COL

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def bss(y,p):
 y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64); bs=float(np.mean((p-y)**2)); return max(0.,100000*(1-bs/(y.mean()*(1-y.mean())))),bs
def fit_cal(y,p):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6); y=np.asarray(y,dtype=np.float64); z=np.log(p/(1-p))
 def obj(ab):
  q=1/(1+np.exp(-np.clip(ab[0]*z+ab[1],-30,30))); return np.mean((q-y)**2)+2e-5*((ab[0]-1)**2+ab[1]**2)
 r=minimize(obj,[1.,0.],method="Nelder-Mead",options={"maxiter":1000,"fatol":1e-12}); return {"type":"logit_affine","a":float(r.x[0]),"b":float(r.x[1])}
def cal(p,c):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6); return 1/(1+np.exp(-np.clip(c["a"]*np.log(p/(1-p))+c["b"],-30,30)))
def params(seed,it):
 return dict(iterations=it,depth=8,learning_rate=.04,loss_function="Logloss",eval_metric="BrierScore",random_seed=seed,l2_leaf_reg=8.,random_strength=.55,bootstrap_type="Bernoulli",subsample=.87,rsm=.88,boosting_type="Ordered",one_hot_max_size=20,max_ctr_complexity=1,thread_count=6,verbose=100,allow_writing_files=False)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--out_dir",default="./model_regime_shared"); a=ap.parse_args(); st=time.time(); path=Path(a.train_path); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(path,encoding="utf-8-sig"); x=build_features(df); y=df[TARGET_COL].to_numpy(np.int8); cats=[c for c in REGIME_CAT_COLS if c in x]; ci=[x.columns.get_loc(c) for c in cats]
 va=df["season"].to_numpy()==2024; tr=~va; xv,yv=x.loc[va],y[va]; h=pd.util.hash_pandas_object(df.loc[va,ID_COL].astype(str),index=False).to_numpy(); stop=(h%4)<2; cf=(h%4)==2; ev=(h%4)==3
 w=(1+.18*(df.loc[tr,"season"].to_numpy()-2019)).astype(np.float32)
 m=CatBoostClassifier(**params(16026,1500)); m.fit(Pool(x.loc[tr],y[tr],cat_features=ci,weight=w),eval_set=Pool(xv.loc[stop],yv[stop],cat_features=ci),early_stopping_rounds=140,use_best_model=True)
 p=m.predict_proba(Pool(xv,cat_features=ci))[:,1]; c=fit_cal(yv[cf],p[cf]); raw=bss(yv[ev],p[ev]); score=bss(yv[ev],cal(p[ev],c)); best=int(m.get_best_iteration()+1)
 print(f"best={best} raw={raw[0]:.3f} calibrated={score[0]:.3f} cal={c}")
 wall=(1+.18*(df["season"].to_numpy()-2019)).astype(np.float32); final=CatBoostClassifier(**params(16026,best)); final.fit(Pool(x,y,cat_features=ci,weight=wall)); final.save_model(out/"regime_shared.cbm")
 art={"schema_version":1,"pipeline":"catboost_game_type_regime_shared_trunk","feature_cols":x.columns.tolist(),"cat_cols":cats,"calibration":c,"predict_threads":6,"validation":{"best_iteration":best,"raw_bss":raw[0],"raw_brier":raw[1],"cal_bss":score[0],"cal_brier":score[1],"split":"2024 hash mod4 stop=0/1 cal=2 untouched=3"},"provenance":{"train_rows":len(df),"train_sha256":sha(path),"python":platform.python_version(),"rules":"official train only; row-wise regime interactions; no test aggregates"}}
 with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False,indent=2)
 print(f"done {time.time()-st:.1f}s -> {out}")
if __name__=="__main__": main()
