#!/usr/bin/env python3
"""최근 3개 시즌 전용 CatBoost: 2021-23→2024 검증, 2022-24→2025 배포."""
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

def params(seed,iterations):
    return dict(iterations=iterations,depth=8,learning_rate=.045,loss_function="Logloss",
        eval_metric="BrierScore",random_seed=seed,l2_leaf_reg=8.0,random_strength=.7,
        bootstrap_type="Bernoulli",subsample=.86,rsm=.88,boosting_type="Ordered",
        one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,
        allow_writing_files=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv")
    ap.add_argument("--out_dir",default="./model_catboost_recent"); ap.add_argument("--iterations",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=3026); a=ap.parse_args(); start=time.time()
    path=Path(a.train_path); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(path,encoding="utf-8-sig"); x=build_features(df); y=df[TARGET_COL].to_numpy(dtype=np.int8)
    cats=[c for c in ALL_CAT_COLS if c in x]; cat_idx=[x.columns.get_loc(c) for c in cats]

    train_mask=df["season"].isin([2021,2022,2023]).to_numpy(); val_mask=(df["season"].to_numpy()==2024)
    xtr,ytr=x.loc[train_mask],y[train_mask]; xva,yva=x.loc[val_mask],y[val_mask]
    hashes=pd.util.hash_pandas_object(df.loc[val_mask,ID_COL].astype(str),index=False).to_numpy()
    stop=(hashes%4)<2; cal_fit=(hashes%4)==2; final_eval=(hashes%4)==3
    train_year=df.loc[train_mask,"season"].to_numpy(); weights=(1+.15*(train_year-2021)).astype(np.float32)
    print(f"validation train 2021-23={len(xtr):,}; 2024 stop/cal/eval={stop.sum():,}/{cal_fit.sum():,}/{final_eval.sum():,}")
    model=CatBoostClassifier(**params(a.seed,a.iterations))
    model.fit(Pool(xtr,ytr,cat_features=cat_idx,weight=weights),
        eval_set=Pool(xva.loc[stop],yva[stop],cat_features=cat_idx),early_stopping_rounds=120,use_best_model=True)
    pred=model.predict_proba(Pool(xva,cat_features=cat_idx))[:,1]
    cal=fit_cal(yva[cal_fit],pred[cal_fit]); raw=bss(yva[final_eval],pred[final_eval]); score=bss(yva[final_eval],calibrate(pred[final_eval],cal))
    best=int(model.get_best_iteration()+1)
    print(f"best={best}, untouched raw={raw[0]:.3f}, calibrated={score[0]:.3f}, cal={cal}")

    deploy_mask=df["season"].isin([2022,2023,2024]).to_numpy(); xd,yd=x.loc[deploy_mask],y[deploy_mask]
    deploy_year=df.loc[deploy_mask,"season"].to_numpy(); deploy_weight=(1+.15*(deploy_year-2022)).astype(np.float32)
    final=CatBoostClassifier(**params(a.seed,best))
    final.fit(Pool(xd,yd,cat_features=cat_idx,weight=deploy_weight)); final.save_model(out/"catboost.cbm")
    art={"schema_version":2,"pipeline":"catboost_recent_three_seasons","feature_cols":x.columns.tolist(),"cat_cols":cats,
        "calibration":cal,"predict_threads":6,
        "training_window":{"validation_train":[2021,2022,2023],"deployment_train":[2022,2023,2024],"annual_weight_step":0.15},
        "validation":{"split":"2024 row_id hash mod4: stop=0/1, calibration=2, untouched=3","best_iteration":best,
            "raw_bss":raw[0],"raw_brier":raw[1],"cal_bss":score[0],"cal_brier":score[1]},
        "provenance":{"source_train_rows":len(df),"deployment_rows":int(deploy_mask.sum()),"train_sha256":sha256(path),
            "seed":a.seed,"python":platform.python_version(),"pandas":pd.__version__,
            "rules":"official train only; fixed recent seasons; test rows independent"}}
    with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False,indent=2)
    print(f"done {time.time()-start:.1f}s -> {out}")

if __name__=="__main__": main()
