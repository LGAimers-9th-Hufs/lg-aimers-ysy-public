#!/usr/bin/env python3
"""공식 TrackMan 시점 안전 피처 CatBoost의 2025용 최종 학습."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from shared.catboost_features import TARGET_COL
from pipelines.trackman.experiments.evaluate_trackman_walkforward import make_features,params
from pipelines.trackman.features import build_artifacts,save_artifacts

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--trackman_path",default="./data/trackman_history.csv"); ap.add_argument("--out_dir",default="./model_trackman"); ap.add_argument("--iterations",type=int,default=187); a=ap.parse_args()
 out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); art=build_artifacts(a.trackman_path); save_artifacts(art,out/"trackman_artifacts.json")
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); y=df[TARGET_COL].to_numpy(np.int8); season=df.season.to_numpy(); x=make_features(df,art); tr=season>=2020
 cats=[c for c in ["top_bottom","game_type","base_state","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","count_state","runner_state","hand_matchup","inning_bucket"] if c in x]; ci=[x.columns.get_loc(c) for c in cats]
 w=(1+.20*(season[tr]-2020)).astype(np.float32); model=CatBoostClassifier(**params(a.iterations)); model.fit(Pool(x.loc[tr],y[tr],cat_features=ci,weight=w)); model.save_model(out/"trackman_catboost.cbm")
 meta={"schema_version":1,"pipeline":"trackman_historical_physics","model_file":"trackman_catboost.cbm","trackman_artifact":"trackman_artifacts.json","iterations":a.iterations,"feature_cols":x.columns.tolist(),"cat_cols":cats,"train_seasons":[2020,2021,2022,2023,2024],"train_sha256":hashlib.sha256(Path(a.train_path).read_bytes()).hexdigest(),"trackman_sha256":hashlib.sha256(Path(a.trackman_path).read_bytes()).hexdigest(),"rules":"official train and 2019-2024 trackman only; row-independent inference; no test aggregates"}
 with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)
 print(json.dumps({"rows":int(tr.sum()),"features":x.shape[1],"iterations":a.iterations},indent=2))
if __name__=="__main__": main()
