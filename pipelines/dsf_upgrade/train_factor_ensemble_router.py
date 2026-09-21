#!/usr/bin/env python3
"""Fit a conditional router from the mean of multiple FactorTabM OOF seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_factor_router import brier, router_features


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--oof_dirs",nargs="+",required=True); ap.add_argument("--output_dir",default="./factor_tabm_ensemble_final"); args=ap.parse_args()
    seeds=[]
    for directory in args.oof_dirs:
        with np.load(Path(directory)/"factor_2023.npz",allow_pickle=False) as saved: seeds.append({key:saved[key] for key in saved.files})
    reference=seeds[0]
    for seed in seeds[1:]:
        if not np.array_equal(seed["row_id"],reference["row_id"]) or not np.array_equal(seed["y"],reference["y"]): raise RuntimeError("OOF alignment mismatch")
    data=reference.copy(); data["factor"]=np.mean([seed["factor"].astype(float) for seed in seeds],axis=0)
    better=((data["y"]-data["factor"])**2 < (data["y"]-data["dsf"])**2).astype(int)
    router=lgb.LGBMClassifier(n_estimators=120,learning_rate=.025,num_leaves=7,max_depth=3,min_child_samples=1500,subsample=.8,colsample_bytree=.9,reg_lambda=20.,verbosity=-1,random_state=44123,n_jobs=6)
    router.fit(router_features(data)[data["stop"]],better[data["stop"]]); route=router.predict_proba(router_features(data))[:,1]
    choices=[]
    for threshold in np.arange(.50,.901,.05):
        for alpha in (.05,.10,.15,.20,.30):
            pred=data["dsf"]+(route>=threshold)*alpha*(data["factor"]-data["dsf"]); choices.append((brier(data["y"][data["select"]],pred[data["select"]]),threshold,alpha))
    _,threshold,alpha=min(choices); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); joblib.dump(router,output/"router.joblib",compress=3)
    (output/"metadata.json").write_text(json.dumps({"threshold":float(threshold),"alpha":float(alpha),"n_seeds":len(seeds),"router_fit":"2023 mean-seed walk-forward OOF stop hash","selection":"2023 select hash only","rules":"official train only; row-local inference"},indent=2),encoding="utf-8")
    print(json.dumps({"threshold":threshold,"alpha":alpha,"n_seeds":len(seeds)}))


if __name__=="__main__": main()
