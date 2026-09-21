#!/usr/bin/env python3
"""Select a stable champion/TrackMan blend with equal season weighting."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
from pipelines.dsf_upgrade.evaluate_factor_router import brier,bss,router_features

def load(directory,year):
    with np.load(Path(directory)/f"factor_{year}.npz",allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--base_oof",default="./factor_tabm_oof");ap.add_argument("--track_oof",default="./factor_tabm_trackman_command_oof");ap.add_argument("--output",default="./trackman_factor_blend.json");args=ap.parse_args()
    base={year:load(args.base_oof,year) for year in (2023,2024)};track={year:load(args.track_oof,year) for year in (2023,2024)};train=base[2023]
    better=((train["y"]-train["factor"])**2<(train["y"]-train["dsf"])**2).astype(int);router=lgb.LGBMClassifier(n_estimators=120,learning_rate=.025,num_leaves=7,max_depth=3,min_child_samples=1500,subsample=.8,colsample_bytree=.9,reg_lambda=20.,verbosity=-1,random_state=44123,n_jobs=6);router.fit(router_features(train)[train["stop"]],better[train["stop"]])
    candidates={}
    for year in (2023,2024):
        d=base[year]
        if not np.array_equal(d["row_id"],track[year]["row_id"]):raise RuntimeError("OOF alignment mismatch")
        route=router.predict_proba(router_features(d))[:,1]>=.5;champion=d["dsf"]+route*.3*(d["factor"]-d["dsf"]);trackman=.8*d["dsf"]+.2*track[year]["factor"];candidates[year]=(champion,trackman)
    def objective(share):
        values=[]
        for year in (2023,2024):
            d=base[year];mask=d["select"];prediction=(1-share)*candidates[year][0]+share*candidates[year][1];reference=d["y"][mask].mean()*(1-d["y"][mask].mean());values.append(brier(d["y"][mask],prediction[mask])/reference)
        return float(np.mean(values))
    grid=np.arange(0,1.01,.05);share=float(min(grid,key=objective));result={"track_candidate_share":share,"track_factor_weight":.2,"router_alpha":.3,"years":{}}
    for year in (2023,2024):
        d=base[year];mask=d["evaluate"];prediction=(1-share)*candidates[year][0]+share*candidates[year][1];result["years"][str(year)]={"delta_bss":bss(d["y"][mask],prediction[mask])-bss(d["y"][mask],d["dsf"][mask])}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
