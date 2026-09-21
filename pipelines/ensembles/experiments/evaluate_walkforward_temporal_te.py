#!/usr/bin/env python3
"""공식 train-only temporal TE의 2022/2023/2024 walk-forward ablation."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from shared.catboost_features import ALL_CAT_COLS, TARGET_COL, build_features

TE_COLS=["pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
PRIORS={"pitcher_id":80.,"batter_id":80.,"pitcher_team_id":800.,"batter_team_id":800.}

def bss(y,p):
    y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
    bs=float(np.mean((p-y)**2)); ref=float(y.mean()*(1-y.mean()))
    return max(0.,100000*(1-bs/ref)),bs

def fit_cal(y,p):
    y=np.asarray(y,dtype=np.float64); p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6); z=np.log(p/(1-p))
    def obj(ab):
        q=1/(1+np.exp(-np.clip(ab[0]*z+ab[1],-30,30)))
        return np.mean((q-y)**2)+2e-5*((ab[0]-1)**2+ab[1]**2)
    r=minimize(obj,[1.,0.],method="Nelder-Mead",options={"maxiter":1000,"fatol":1e-12})
    return {"a":float(r.x[0]),"b":float(r.x[1])}

def calibrate(p,c):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
    return 1/(1+np.exp(-np.clip(c["a"]*np.log(p/(1-p))+c["b"],-30,30)))

def maps(history, fallback, prior_scale=1.):
    out={}
    for c in TE_COLS:
        stat=history.groupby(c,observed=True)[TARGET_COL].agg(["sum","count"])
        m=PRIORS[c]*prior_scale
        out[c]=((stat["sum"]+m*fallback)/(stat["count"]+m)).to_dict()
    return out

def add_lookup(x,rows,all_maps,recent_maps,fallback):
    z=x.copy()
    for c in TE_COLS:
        key=rows[c]
        z[f"hist_te_{c}"]=key.map(all_maps[c]).fillna(fallback).astype(np.float32)
        z[f"recent_te_{c}"]=key.map(recent_maps[c]).fillna(z[f"hist_te_{c}"]).astype(np.float32)
    z["hist_te_player_diff"]=(z["hist_te_pitcher_id"]-z["hist_te_batter_id"]).astype(np.float32)
    z["recent_te_player_diff"]=(z["recent_te_pitcher_id"]-z["recent_te_batter_id"]).astype(np.float32)
    z["pitcher_te_momentum"]=(z["recent_te_pitcher_id"]-z["hist_te_pitcher_id"]).astype(np.float32)
    z["batter_te_momentum"]=(z["recent_te_batter_id"]-z["hist_te_batter_id"]).astype(np.float32)
    return z

def temporal_training_features(base,df,train_years):
    pieces=[]
    for season in train_years:
        idx=df.index[df["season"]==season]; rows=df.loc[idx]; history=df[df["season"]<season]
        fallback=float(history[TARGET_COL].mean()) if len(history) else .5
        if len(history):
            all_m=maps(history,fallback); recent=history[history["season"]==history["season"].max()]; recent_m=maps(recent,fallback,.75)
        else:
            all_m={c:{} for c in TE_COLS}; recent_m={c:{} for c in TE_COLS}
        part=add_lookup(base.loc[idx],rows,all_m,recent_m,fallback); pieces.append(part)
    return pd.concat(pieces).sort_index()

def validation_features(base,df,year):
    idx=df.index[df["season"]==year]; history=df[df["season"]<year]; fallback=float(history[TARGET_COL].mean())
    all_m=maps(history,fallback); recent=history[history["season"]==year-1]; recent_m=maps(recent,fallback,.75)
    return add_lookup(base.loc[idx],df.loc[idx],all_m,recent_m,fallback)

def params(seed,iterations):
    return dict(iterations=iterations,depth=8,learning_rate=.045,loss_function="Logloss",eval_metric="BrierScore",
        random_seed=seed,l2_leaf_reg=7.,random_strength=.6,bootstrap_type="Bernoulli",subsample=.85,rsm=.85,
        boosting_type="Ordered",one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,
        allow_writing_files=False)

def fit_predict(xtr,ytr,xva,cat_names,weights,seed,iterations):
    cat_idx=[xtr.columns.get_loc(c) for c in cat_names]
    model=CatBoostClassifier(**params(seed,iterations))
    model.fit(Pool(xtr,ytr,cat_features=cat_idx,weight=weights))
    return model.predict_proba(Pool(xva,cat_features=cat_idx))[:,1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv")
    ap.add_argument("--out",default="./walkforward_temporal_te.json"); ap.add_argument("--iterations",type=int,default=193)
    a=ap.parse_args(); start=time.time(); df=pd.read_csv(a.train_path,encoding="utf-8-sig"); base=build_features(df)
    y=df[TARGET_COL].to_numpy(dtype=np.int8); results={"baseline":{},"temporal_te":{}}; stored={"baseline":{},"temporal_te":{}}
    for year in [2022,2023,2024]:
        train_years=sorted(df.loc[df["season"]<year,"season"].unique().tolist())
        tr=df["season"].to_numpy()<year; va=df["season"].to_numpy()==year
        w=(1+.18*(df.loc[tr,"season"].to_numpy()-2019)).astype(np.float32)
        xbtr,xbva=base.loc[tr],base.loc[va]; cats=[c for c in ALL_CAT_COLS if c in xbtr]
        print(f"\n[{year}] baseline train={tr.sum():,} val={va.sum():,}",flush=True)
        pb=fit_predict(xbtr,y[tr],xbva,cats,w,2026+year,a.iterations)
        print(f"[{year}] temporal TE build",flush=True)
        xttr=temporal_training_features(base,df,train_years); xtva=validation_features(base,df,year)
        print(f"[{year}] temporal TE fit features={xttr.shape[1]}",flush=True)
        pt=fit_predict(xttr,y[tr],xtva,cats,w,3026+year,a.iterations)
        for name,p in [("baseline",pb),("temporal_te",pt)]:
            stored[name][year]=p; raw=bss(y[va],p)
            results[name][str(year)]={"raw_bss":raw[0],"raw_brier":raw[1]}
            print(f"[{year}] {name} raw BSS={raw[0]:.3f}")

    # Calibration is always learned from the immediately previous OOF season.
    for name in ["baseline","temporal_te"]:
        fold_scores=[]
        for year,cal_year in [(2023,2022),(2024,2023)]:
            ycal=df.loc[df["season"]==cal_year,TARGET_COL].to_numpy(); yev=df.loc[df["season"]==year,TARGET_COL].to_numpy()
            cal=fit_cal(ycal,stored[name][cal_year]); score=bss(yev,calibrate(stored[name][year],cal))
            results[name][str(year)]["prior_year_cal_bss"]=score[0]
            results[name][str(year)]["prior_year_cal_brier"]=score[1]; fold_scores.append(score[0])
        results[name]["summary"]={"raw_mean_bss":float(np.mean([results[name][str(y)]["raw_bss"] for y in [2022,2023,2024]])),
            "raw_std_bss":float(np.std([results[name][str(y)]["raw_bss"] for y in [2022,2023,2024]])),
            "prior_year_cal_mean_bss_2023_2024":float(np.mean(fold_scores)),"prior_year_cal_worst_bss":float(np.min(fold_scores))}
    results["delta"]={k:results["temporal_te"]["summary"][k]-results["baseline"]["summary"][k] for k in results["baseline"]["summary"]}
    results["iterations"]=a.iterations; results["elapsed_seconds"]=time.time()-start
    results["rules"]="official train labels from strictly prior seasons only; no test or leaderboard information"
    with open(a.out,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
