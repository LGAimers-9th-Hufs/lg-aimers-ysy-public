#!/usr/bin/env python3
"""전체기간/최근3시즌 CatBoost의 strict 2024 OOF blend 평가."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.optimize import minimize, minimize_scalar
from shared.catboost_features import ALL_CAT_COLS, ID_COL, TARGET_COL, build_features

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

def full_params(iterations):
    return dict(iterations=iterations,depth=8,learning_rate=.045,loss_function="Logloss",eval_metric="BrierScore",
        random_seed=2026,l2_leaf_reg=7.,random_strength=.6,bootstrap_type="Bernoulli",subsample=.85,rsm=.85,
        boosting_type="Ordered",one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,allow_writing_files=False)

def recent_params(iterations):
    return dict(iterations=iterations,depth=8,learning_rate=.045,loss_function="Logloss",eval_metric="BrierScore",
        random_seed=3026,l2_leaf_reg=8.,random_strength=.7,bootstrap_type="Bernoulli",subsample=.86,rsm=.88,
        boosting_type="Ordered",one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,allow_writing_files=False)

def brier_params(iterations):
    return dict(iterations=iterations,depth=8,learning_rate=.04,loss_function="RMSE",eval_metric="RMSE",
        random_seed=7048,l2_leaf_reg=9.,random_strength=.5,bootstrap_type="Bernoulli",subsample=.86,rsm=.88,
        boosting_type="Ordered",one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,allow_writing_files=False)

def fit_oof(name,x,y,df,train_mask,xva,yva,stop,cat_idx,weights,params):
    print(f"[{name}] train rows={train_mask.sum():,}")
    model=CatBoostClassifier(**params(1000))
    model.fit(Pool(x.loc[train_mask],y[train_mask],cat_features=cat_idx,weight=weights),
        eval_set=Pool(xva.loc[stop],yva[stop],cat_features=cat_idx),early_stopping_rounds=120,use_best_model=True)
    pred=model.predict_proba(Pool(xva,cat_features=cat_idx))[:,1]
    print(f"[{name}] best={model.get_best_iteration()+1}")
    return pred,int(model.get_best_iteration()+1)

def fit_brier_oof(x,y,train_mask,xva,yva,stop,cat_idx,weights):
    print(f"[brier] train rows={train_mask.sum():,}")
    model=CatBoostRegressor(**brier_params(1200))
    model.fit(Pool(x.loc[train_mask],y[train_mask],cat_features=cat_idx,weight=weights),
        eval_set=Pool(xva.loc[stop],yva[stop],cat_features=cat_idx),early_stopping_rounds=120,use_best_model=True)
    pred=np.clip(model.predict(Pool(xva,cat_features=cat_idx)),0,1)
    print(f"[brier] best={model.get_best_iteration()+1}")
    return pred,int(model.get_best_iteration()+1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv")
    ap.add_argument("--out",default="./catboost_blend_oof.json"); a=ap.parse_args(); start=time.time()
    df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=build_features(df); y=df[TARGET_COL].to_numpy(dtype=np.int8)
    cats=[c for c in ALL_CAT_COLS if c in x]; cat_idx=[x.columns.get_loc(c) for c in cats]
    val=df["season"].to_numpy()==2024; xva,yva=x.loc[val],y[val]
    hashes=pd.util.hash_pandas_object(df.loc[val,ID_COL].astype(str),index=False).to_numpy()
    stop=(hashes%4)<2; cal_fit=(hashes%4)==2; final_eval=(hashes%4)==3

    full_mask=df["season"].to_numpy()<2024
    full_year=df.loc[full_mask,"season"].to_numpy(); full_w=(1+.18*(full_year-2019)).astype(np.float32)
    p_full,it_full=fit_oof("full",x,y,df,full_mask,xva,yva,stop,cat_idx,full_w,full_params)
    recent_mask=df["season"].isin([2021,2022,2023]).to_numpy()
    recent_year=df.loc[recent_mask,"season"].to_numpy(); recent_w=(1+.15*(recent_year-2021)).astype(np.float32)
    p_recent,it_recent=fit_oof("recent3",x,y,df,recent_mask,xva,yva,stop,cat_idx,recent_w,recent_params)
    p_brier,it_brier=fit_brier_oof(x,y,full_mask,xva,yva,stop,cat_idx,full_w)

    matrix=np.column_stack([p_full,p_recent,p_brier])
    def objective(w): return float(np.mean((matrix[cal_fit]@w-yva[cal_fit])**2))
    opt=minimize(objective,np.full(3,1/3),method="SLSQP",bounds=[(0.,1.)]*3,
        constraints={"type":"eq","fun":lambda w:w.sum()-1.},options={"maxiter":500,"ftol":1e-14})
    weights=np.clip(np.asarray(opt.x,dtype=np.float64),0,1); weights/=weights.sum(); blend=matrix@weights
    cal=fit_cal(yva[cal_fit],blend[cal_fit])
    results={
        "split":"2024 row_id hash mod4: stop=0/1, blend+calibration=2, untouched=3",
        "n_stop":int(stop.sum()),"n_cal":int(cal_fit.sum()),"n_eval":int(final_eval.sum()),
        "best_iterations":{"full":it_full,"recent3":it_recent,"brier":it_brier},
        "weights":{"full":float(weights[0]),"recent3":float(weights[1]),"brier":float(weights[2])},"calibration":cal,
        "full_raw_eval":bss(yva[final_eval],p_full[final_eval]),
        "recent_raw_eval":bss(yva[final_eval],p_recent[final_eval]),
        "brier_raw_eval":bss(yva[final_eval],p_brier[final_eval]),
        "blend_raw_eval":bss(yva[final_eval],blend[final_eval]),
        "blend_cal_eval":bss(yva[final_eval],calibrate(blend[final_eval],cal)),
        "prediction_correlation_eval":np.corrcoef(matrix[final_eval],rowvar=False).tolist(),
        "elapsed_seconds":time.time()-start,
        "rules":"official train only; no test or leaderboard information"
    }
    with open(a.out,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
