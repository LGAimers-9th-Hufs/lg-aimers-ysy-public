#!/usr/bin/env python3
"""독립 CatBoost ordered-boosting 학습 파이프라인."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from shared.catboost_features import ALL_CAT_COLS, ID_COL, TARGET_COL, build_features

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def bss(y,p):
    y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
    bs=float(np.mean((p-y)**2)); ref=float(y.mean()*(1-y.mean()))
    return max(0.,100000*(1-bs/ref)),bs

def fit_cal(y,p):
    y=np.asarray(y,dtype=np.float64); p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
    z=np.log(p/(1-p))
    def obj(ab):
        out=1/(1+np.exp(-np.clip(ab[0]*z+ab[1],-30,30)))
        return np.mean((out-y)**2)+2e-5*((ab[0]-1)**2+ab[1]**2)
    r=minimize(obj,[1.,0.],method="Nelder-Mead",options={"maxiter":1000,"fatol":1e-12})
    return {"type":"logit_affine","a":float(r.x[0]),"b":float(r.x[1])}

def calibrate(p,c):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
    z=np.clip(c["a"]*np.log(p/(1-p))+c["b"],-30,30)
    return 1/(1+np.exp(-z))

def model_params(seed,iterations):
    return dict(iterations=iterations,depth=8,learning_rate=.045,loss_function="Logloss",
        eval_metric="BrierScore",random_seed=seed,l2_leaf_reg=7.0,random_strength=.6,
        bootstrap_type="Bernoulli",subsample=.85,rsm=.85,boosting_type="Ordered",
        one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,
        allow_writing_files=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv")
    ap.add_argument("--out_dir",default="./model_catboost"); ap.add_argument("--iterations",type=int,default=1200)
    ap.add_argument("--seed",type=int,default=2026); a=ap.parse_args(); start=time.time()
    train_path=Path(a.train_path); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(train_path,encoding="utf-8-sig"); x=build_features(df)
    cats=[c for c in ALL_CAT_COLS if c in x.columns]; cat_idx=[x.columns.get_loc(c) for c in cats]
    y=df[TARGET_COL].to_numpy(dtype=np.int8); val=df["season"].to_numpy()==2024
    hashes=pd.util.hash_pandas_object(df.loc[val,ID_COL].astype(str),index=False).to_numpy()
    stop=(hashes%4)<2; cal_fit=(hashes%4)==2; final_eval=(hashes%4)==3
    xtr,xva=x.loc[~val],x.loc[val]; ytr,yva=y[~val],y[val]
    weights=(1+.18*(df.loc[~val,"season"].to_numpy()-2019)).astype(np.float32)
    tr_pool=Pool(xtr,ytr,cat_features=cat_idx,weight=weights)
    stop_pool=Pool(xva.loc[stop],yva[stop],cat_features=cat_idx)
    print(f"train={len(xtr):,}, features={x.shape[1]}, cats={len(cats)}, stop/cal/eval={stop.sum():,}/{cal_fit.sum():,}/{final_eval.sum():,}")
    model=CatBoostClassifier(**model_params(a.seed,a.iterations))
    model.fit(tr_pool,eval_set=stop_pool,early_stopping_rounds=120,use_best_model=True)
    pred=model.predict_proba(Pool(xva,cat_features=cat_idx))[:,1]
    cal=fit_cal(yva[cal_fit],pred[cal_fit]); raw=bss(yva[final_eval],pred[final_eval]); calibrated=bss(yva[final_eval],calibrate(pred[final_eval],cal))
    best=int(model.get_best_iteration()+1); print(f"best={best}, untouched raw={raw[0]:.3f}, calibrated={calibrated[0]:.3f}, cal={cal}")
    full_weights=(1+.18*(df["season"].to_numpy()-2019)).astype(np.float32)
    final=CatBoostClassifier(**model_params(a.seed,best)); final.fit(Pool(x,y,cat_features=cat_idx,weight=full_weights))
    final.save_model(out/"catboost.cbm")
    art={"schema_version":1,"pipeline":"catboost_ordered","feature_cols":x.columns.tolist(),"cat_cols":cats,
         "calibration":cal,"predict_threads":6,
         "validation":{"split":"2024 row_id hash mod4: stop=0/1, calibration=2, untouched=3","best_iteration":best,
          "raw_bss":raw[0],"raw_brier":raw[1],"cal_bss":calibrated[0],"cal_brier":calibrated[1]},
         "provenance":{"train_rows":len(df),"train_sha256":sha256(train_path),"seed":a.seed,
          "python":platform.python_version(),"pandas":pd.__version__,"rules":"official train only; test rows independent"}}
    with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False,indent=2)
    print(f"done {time.time()-start:.1f}s -> {out}")

if __name__=="__main__": main()
