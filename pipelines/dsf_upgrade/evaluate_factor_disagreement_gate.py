#!/usr/bin/env python3
"""Evaluate a seed-disagreement safety gate using only walk-forward OOF."""
from __future__ import annotations

import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_factor_router import brier,bss,router_features


def load(directory,year):
    with np.load(Path(directory)/f"factor_{year}.npz",allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--oof_dirs",nargs="+",required=True);ap.add_argument("--output",default="./factor_disagreement_gate.json");args=ap.parse_args()
    data={}
    for year in (2023,2024):
        seeds=[load(directory,year) for directory in args.oof_dirs];reference=seeds[0]
        for seed in seeds[1:]:
            if not np.array_equal(seed["row_id"],reference["row_id"]):raise RuntimeError("OOF alignment mismatch")
        block=reference.copy();block["factors"]=np.column_stack([seed["factor"] for seed in seeds]);data[year]=block
    train=data[2023];better=((train["y"]-train["factor"])**2<(train["y"]-train["dsf"])**2).astype(int)
    router=lgb.LGBMClassifier(n_estimators=120,learning_rate=.025,num_leaves=7,max_depth=3,min_child_samples=1500,subsample=.8,colsample_bytree=.9,reg_lambda=20.,verbosity=-1,random_state=44123,n_jobs=6)
    router.fit(router_features(train)[train["stop"]],better[train["stop"]])
    block=data[2024];route=router.predict_proba(router_features(block))[:,1]>=.5;dispersion=np.std(block["factors"],axis=1)
    reference_direction=np.sign(block["factor"]-block["dsf"]);agreement=(np.sign(block["factors"]-block["dsf"][:,None])==reference_direction[:,None]).sum(1)
    choices=[]
    for percentile in (20,30,40,50,60,70,80,90,95,100):
        cutoff=np.percentile(dispersion[block["select"]],percentile)
        for required in (1,2,3):
            gate=(dispersion<=cutoff)&(agreement>=required);prediction=block["dsf"]+route*gate*.3*(block["factor"]-block["dsf"])
            choices.append((brier(block["y"][block["select"]],prediction[block["select"]]),percentile,float(cutoff),required))
    _,percentile,cutoff,required=min(choices);gate=(dispersion<=cutoff)&(agreement>=required);single=block["dsf"]+route*.3*(block["factor"]-block["dsf"]);prediction=block["dsf"]+route*gate*.3*(block["factor"]-block["dsf"]);mask=block["evaluate"]
    result={"percentile":percentile,"cutoff":cutoff,"required_agreement":required,"single_delta_bss":bss(block["y"][mask],single[mask])-bss(block["y"][mask],block["dsf"][mask]),"gate_delta_bss":bss(block["y"][mask],prediction[mask])-bss(block["y"][mask],block["dsf"][mask]),"routed_fraction":float(np.mean((route&gate)[mask])),"accepted":False}
    result["accepted"]=bool(result["gate_delta_bss"]>=5.0);Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
