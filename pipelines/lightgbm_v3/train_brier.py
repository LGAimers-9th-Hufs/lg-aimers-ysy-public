#!/usr/bin/env python3
"""V3 동일 피처의 직접 Brier(L2) 보조 모델 최종 학습."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from pipelines.lightgbm_v3.train import base_artifact,temporal_features
from pipelines.lightgbm_v3.experiments.evaluate_v3_brier import params

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--out_dir",default="./model_v3_brier"); ap.add_argument("--iterations",type=int,default=315); a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=temporal_features(df); y=df.control_success.to_numpy(np.float32); w=(1+.18*(df.season.to_numpy()-2019)).astype(np.float32); p=params(10326); p["n_estimators"]=a.iterations; m=lgb.LGBMRegressor(**p); m.fit(x,y,sample_weight=w); m.booster_.save_model(str(out/"v3_brier.txt"))
 art=base_artifact(df,df); art.update({"model_file":"v3_brier.txt","feature_cols":x.columns.tolist(),"iterations":a.iterations,"calibration":{"type":"logit_affine","a":1.0687266425200987,"b":-0.051549934240265},"predict_threads":6,"rules":"official train only; row-independent inference; no test aggregates"})
 with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False,indent=2)
 print(json.dumps({"rows":len(df),"features":x.shape[1],"iterations":a.iterations},indent=2))
if __name__=="__main__": main()
