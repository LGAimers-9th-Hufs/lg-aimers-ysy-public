#!/usr/bin/env python3
"""Fit the frozen conditional router in a LightGBM-only process."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from pipelines.dsf_upgrade.evaluate_factor_router import router_features


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--factor_oof_2023",default="./factor_tabm_oof/factor_2023.npz"); ap.add_argument("--output_dir",default="./factor_tabm_final"); args=ap.parse_args()
    with np.load(args.factor_oof_2023,allow_pickle=False) as saved: oof={key:saved[key] for key in saved.files}
    better=((oof["y"]-oof["factor"])**2 < (oof["y"]-oof["dsf"])**2).astype(int)
    router=lgb.LGBMClassifier(n_estimators=120,learning_rate=.025,num_leaves=7,max_depth=3,min_child_samples=1500,subsample=.8,colsample_bytree=.9,reg_lambda=20.,verbosity=-1,random_state=44123,n_jobs=6)
    router.fit(router_features(oof)[oof["stop"]],better[oof["stop"]]); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); joblib.dump(router,output/"router.joblib",compress=3)
    print(f"saved {output/'router.joblib'}")


if __name__=="__main__": main()
