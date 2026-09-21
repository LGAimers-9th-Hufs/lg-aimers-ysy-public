#!/usr/bin/env python3
"""시점 안전 TrackMan 물리 프로파일 CatBoost의 strict 2024 OOF 평가."""
from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from shared.catboost_features import ID_COL,TARGET_COL,build_features
from pipelines.catboost.experiments.evaluate_catboost_blend import bss,fit_cal,calibrate
from pipelines.trackman.features import build_artifacts,enrich_trackman,save_artifacts

DROP_DIVERSITY={"pitcher_id","batter_id","team_matchup"}

def params(iterations=1200):
 return dict(iterations=iterations,depth=8,learning_rate=.04,loss_function="Logloss",eval_metric="BrierScore",
  random_seed=86061,l2_leaf_reg=10.,random_strength=.8,bootstrap_type="Bernoulli",subsample=.84,rsm=.9,
  boosting_type="Ordered",one_hot_max_size=16,max_ctr_complexity=1,thread_count=6,verbose=100,allow_writing_files=False)

def make_features(df,art):
 x=build_features(df); x=x.drop(columns=[c for c in DROP_DIVERSITY if c in x]); tm=enrich_trackman(df,art)
 for c in tm: x[c]=tm[c]
 p=pd.to_numeric(df["asof_pitcher_success_rate"],errors="coerce").fillna(.5)
 x["tm_skill_x_speed"]=(p*x["tm_exp_rel_speed"]).astype(np.float32)
 x["tm_skill_x_movement"]=(p*x["tm_movement_separation"]).astype(np.float32)
 x["tm_spin_x_breaking"]=(x["tm_exp_spin_rate"]*pd.to_numeric(df["asof_pitcher_breaking_rate"],errors="coerce").fillna(0)).astype(np.float32)
 return x

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--trackman_path",default="./data/trackman_history.csv")
 ap.add_argument("--out",default="./trackman_oof.json"); ap.add_argument("--pred_out",default="./trackman_2024_oof.npz"); ap.add_argument("--artifact_out",default="./trackman_artifacts.json")
 a=ap.parse_args(); st=time.time(); art=build_artifacts(a.trackman_path); save_artifacts(art,a.artifact_out)
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); y=df[TARGET_COL].to_numpy(np.int8); season=df["season"].to_numpy(); x=make_features(df,art)
 cats=[c for c in ["top_bottom","game_type","base_state","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","count_state","runner_state","hand_matchup","inning_bucket"] if c in x]
 ci=[x.columns.get_loc(c) for c in cats]; tr=(season>=2020)&(season<2024); va=season==2024; xv=x.loc[va]; yv=y[va]
 h=pd.util.hash_pandas_object(df.loc[va,ID_COL].astype(str),index=False).to_numpy(); stop=(h%4)<2; cf=(h%4)==2; ev=(h%4)==3
 w=(1+.20*(season[tr]-2020)).astype(np.float32); model=CatBoostClassifier(**params())
 model.fit(Pool(x.loc[tr],y[tr],cat_features=ci,weight=w),eval_set=Pool(xv.loc[stop],yv[stop],cat_features=ci),early_stopping_rounds=140,use_best_model=True)
 pred=model.predict_proba(Pool(xv,cat_features=ci))[:,1]; cal=fit_cal(yv[cf],pred[cf]); pc=calibrate(pred,cal)
 result={"best_iteration":model.get_best_iteration()+1,"n_features":x.shape[1],"feature_cols":x.columns.tolist(),"cat_cols":cats,
  "raw_eval":bss(yv[ev],pred[ev]),"calibration":cal,"cal_eval":bss(yv[ev],pc[ev]),"elapsed_seconds":time.time()-st,
  "rules":"official train + trackman only; TrackMan seasons strictly before each row season; test rows independent"}
 with open(a.out,"w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
 np.savez_compressed(a.pred_out,row_id=df.loc[va,ID_COL].astype(str).to_numpy(dtype=object),y=yv,pred=pred,stop=stop,cal_fit=cf,eval=ev)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
