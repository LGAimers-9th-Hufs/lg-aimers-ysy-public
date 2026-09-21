#!/usr/bin/env python3
"""선택이 끝난 안정 피처로 2019--2024 최종 CatBoost 두 개를 학습한다."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from shared.catboost_features import TARGET_COL, build_features
from pipelines.catboost.experiments.evaluate_catboost_blend import full_params, recent_params

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv")
    ap.add_argument("--selection",required=True); ap.add_argument("--model_dir",default="./model_stable_features")
    a=ap.parse_args(); out=Path(a.model_dir); out.mkdir(parents=True,exist_ok=True)
    with open(a.selection,encoding="utf-8") as f: sel=json.load(f)
    df=pd.read_csv(a.train_path,encoding="utf-8-sig"); y=df[TARGET_COL].to_numpy(np.int8)
    x=build_features(df)[sel["feature_cols"]]; ci=[x.columns.get_loc(c) for c in sel["cat_cols"]]
    seasons=df["season"].to_numpy(); itf=int(sel["best_iterations"]["full"]); itr=int(sel["best_iterations"]["recent3"])
    fy=seasons; fw=(1+.18*(fy-2019)).astype(np.float32)
    full=CatBoostClassifier(**full_params(itf)); full.fit(Pool(x,y,cat_features=ci,weight=fw)); full.save_model(out/"catboost_full.cbm")
    recent=seasons>=2022; ry=seasons[recent]; rw=(1+.15*(ry-2022)).astype(np.float32)
    rec=CatBoostClassifier(**recent_params(itr)); rec.fit(Pool(x.loc[recent],y[recent],cat_features=ci,weight=rw)); rec.save_model(out/"catboost_recent3.cbm")
    art={"schema_version":1,"pipeline":"walkforward_fold_stable_features","model_files":["catboost_full.cbm","catboost_recent3.cbm"],
        "weights":sel["weights"],"feature_cols":sel["feature_cols"],"cat_cols":sel["cat_cols"],"calibration":sel["calibration"],"predict_threads":6,
        "validation":{"selection_folds":sel["selection_folds"],"best_iterations":sel["best_iterations"],"blend_cal_eval":sel["blend_cal_eval"]},
        "provenance":{"train_rows":len(df),"train_sha256":hashlib.sha256(Path(a.train_path).read_bytes()).hexdigest(),
        "rules":"official train only; feature selection through 2023; no test aggregates or external data"}}
    with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False,indent=2)
    print(json.dumps(art,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
