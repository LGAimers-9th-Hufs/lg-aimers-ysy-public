#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3: 최근 시즌 가중과 Brier 최적 블렌딩, strict three-way 2024 validation."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pipelines.lightgbm_v2.inference import build_features

ID_COL,TARGET_COL="row_id","control_success"
TE_COLS=["pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
TE_PRIORS={"pitcher_id":80.0,"batter_id":80.0,"pitcher_team_id":800.0,"batter_team_id":800.0}
CAT_COLS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
RAW_NUMERIC_COLS=[
 "season","game_month","game_dayofweek","inning","balls_before","strikes_before","outs_before",
 "run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team",
 "runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","home_win_expectancy",
 "away_win_expectancy","li","asof_pitcher_n","asof_pitcher_success_rate",
 "asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate",
 "asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate",
 "asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate",
 "asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate",
 "asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate",
 "asof_batter_middle_rate","asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate",
 "asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]

def file_sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
 return h.hexdigest()

def bss(y,p):
 y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
 bs=float(np.mean((p-y)**2)); ref=float(y.mean()*(1-y.mean()))
 return max(0.0,100000*(1-bs/ref)),bs

def make_te_maps(df,fallback):
 maps={}
 for c in TE_COLS:
  stat=df.groupby(c,observed=True)[TARGET_COL].agg(["sum","count"])
  m=TE_PRIORS[c]; values=(stat["sum"]+m*fallback)/(stat["count"]+m)
  maps[c]={str(k):float(v) for k,v in values.items()}
 return maps

def base_artifact(train,history):
 fallback=float(history[TARGET_COL].mean()) if len(history) else .5
 cat_maps={}
 for c in CAT_COLS:
  values=sorted(train[c].dropna().astype(str).unique().tolist())
  cat_maps[c]={v:i for i,v in enumerate(values)}
 return {"schema_version":3,"global_mean":fallback,"raw_numeric_cols":RAW_NUMERIC_COLS,
  "cat_maps":cat_maps,"te_cols":TE_COLS,
  "te_maps":make_te_maps(history,fallback) if len(history) else {c:{} for c in TE_COLS},
  "required_input_cols":sorted(set(RAW_NUMERIC_COLS+CAT_COLS+TE_COLS))}

def temporal_features(df):
 pieces=[]
 for season in sorted(df["season"].unique()):
  rows=df[df["season"]==season]; history=df[df["season"]<season]
  part=build_features(rows,base_artifact(df,history)); part.index=rows.index; pieces.append(part)
  print(f"  season {season}: rows={len(rows):,}, history={len(history):,}",flush=True)
 return pd.concat(pieces).sort_index()

def params(seed,variant,rounds):
 configs=[
  dict(num_leaves=63,max_depth=8,min_child_samples=140,learning_rate=.025),
  dict(num_leaves=95,max_depth=10,min_child_samples=220,learning_rate=.020),
  dict(num_leaves=47,max_depth=7,min_child_samples=100,learning_rate=.030)]
 return dict(objective="binary",metric="binary_logloss",boosting_type="gbdt",
  n_estimators=rounds,feature_fraction=.86,bagging_fraction=.88,bagging_freq=1,
  reg_alpha=.2,reg_lambda=2.0,verbosity=-1,n_jobs=6,seed=seed,
  deterministic=True,force_col_wise=True,cat_l2=12.0,cat_smooth=18.0,max_cat_to_onehot=16,**configs[variant])

def fit_calibration(y,p):
 y=np.asarray(y,dtype=np.float64); p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
 logit=np.log(p/(1-p))
 def objective(ab):
  out=1/(1+np.exp(-np.clip(ab[0]*logit+ab[1],-30,30)))
  return np.mean((out-y)**2)+2e-5*(ab[0]-1)**2+2e-5*ab[1]**2
 res=minimize(objective,[1.,0.],method="Nelder-Mead",options={"maxiter":1000,"xatol":1e-8,"fatol":1e-12})
 return {"type":"logit_affine","a":float(res.x[0]),"b":float(res.x[1])}

def calibrate(p,cal):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
 z=np.clip(cal["a"]*np.log(p/(1-p))+cal["b"],-30,30)
 return 1/(1+np.exp(-z))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv")
 ap.add_argument("--out_dir",default="./model_v3"); ap.add_argument("--seed",type=int,default=142)
 ap.add_argument("--max_rounds",type=int,default=1800); args=ap.parse_args()
 start=time.time(); train_path=Path(args.train_path); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
 train=pd.read_csv(train_path,encoding="utf-8-sig")
 if train[TARGET_COL].isna().any() or not set(train[TARGET_COL].unique()).issubset({0,1}): raise ValueError("target 오류")
 print(f"rows={len(train):,}, seasons={sorted(train.season.unique())}, mean={train[TARGET_COL].mean():.6f}")
 x=temporal_features(train); feature_cols=x.columns.tolist(); y=train[TARGET_COL].to_numpy(dtype=np.float32)
 is_val=train["season"].to_numpy()==2024; xtr,ytr=x.loc[~is_val],y[~is_val]; xva,yva=x.loc[is_val],y[is_val]
 hashes=pd.util.hash_pandas_object(train.loc[is_val,ID_COL].astype(str),index=False).to_numpy()
 stop_mask=(hashes%4)<2; cal_mask=(hashes%4)==2; eval_mask=(hashes%4)==3
 wtr=(1+.18*(train.loc[~is_val,"season"].to_numpy()-2019)).astype(np.float32)
 # V2처럼 정수 인코딩을 수치로 처리한다. native categorical은 시즌 이동 검증에서 과적합했다.
 native_cat=[]
 print(f"features={len(feature_cols)}, categorical={len(native_cat)}, train={len(xtr):,}, stop/cal/eval={stop_mask.sum():,}/{cal_mask.sum():,}/{eval_mask.sum():,}")
 val_preds=[]; best_rounds=[]
 for i in range(3):
  model=lgb.LGBMClassifier(**params(args.seed+i*97,i,args.max_rounds))
  model.fit(xtr,ytr,sample_weight=wtr,categorical_feature=native_cat,
   eval_set=[(xva.loc[stop_mask],yva[stop_mask])],callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(250)])
  pred=model.predict_proba(xva)[:,1]; val_preds.append(pred); best_rounds.append(int(model.best_iteration_ or args.max_rounds))
  print(f"model{i}: rounds={best_rounds[-1]}, untouched_eval_BSS={bss(yva[eval_mask],pred[eval_mask])[0]:.3f}")
 pred_matrix=np.column_stack(val_preds)
 def blend_objective(w):
  return float(np.mean((pred_matrix[cal_mask]@w-yva[cal_mask])**2))
 blend_opt=minimize(blend_objective,np.full(3,1/3),method="SLSQP",
  bounds=[(0.,1.)]*3,constraints={"type":"eq","fun":lambda w:w.sum()-1.},
  options={"maxiter":500,"ftol":1e-14})
 weights=np.asarray(blend_opt.x,dtype=np.float64)
 weights=np.clip(weights,0,1); weights/=weights.sum()
 blend=pred_matrix@weights
 cal=fit_calibration(yva[cal_mask],blend[cal_mask])
 raw_score=bss(yva[eval_mask],blend[eval_mask]); cal_score=bss(yva[eval_mask],calibrate(blend[eval_mask],cal))
 print(f"weights={weights.tolist()}")
 print(f"UNTOUCHED eval raw={raw_score[0]:.3f}, calibrated={cal_score[0]:.3f}, cal={cal}")
 wall=(1+.18*(train["season"].to_numpy()-2019)).astype(np.float32); model_files=[]
 for i,rounds in enumerate(best_rounds):
  model=lgb.LGBMClassifier(**params(args.seed+i*97,i,rounds))
  model.fit(x,y,sample_weight=wall,categorical_feature=native_cat)
  name=f"lgb{i}.txt"; model.booster_.save_model(str(out/name)); model_files.append(name)
  print(f"saved {name}: rounds={rounds}")
 art=base_artifact(train,train)
 art.update({"feature_cols":feature_cols,"categorical_features":native_cat,"model_files":model_files,
  "blend_weights":weights.tolist(),"calibration":cal,"predict_threads":6,
  "validation":{"season":2024,"split":"row_id hash mod4: stop=0/1, calibration=2, untouched_eval=3",
   "n_stop":int(stop_mask.sum()),"n_cal":int(cal_mask.sum()),"n_eval":int(eval_mask.sum()),
   "raw_bss_eval":float(raw_score[0]),"raw_brier_eval":float(raw_score[1]),
   "cal_bss_eval":float(cal_score[0]),"cal_brier_eval":float(cal_score[1]),"best_rounds":best_rounds},
  "provenance":{"train_rows":int(len(train)),"train_sha256":file_sha256(train_path),"seed":args.seed,
   "python":platform.python_version(),"lightgbm":lgb.__version__,"pandas":pd.__version__,"numpy":np.__version__,
   "rules":"test rows independent; official train-only maps; no test-derived aggregates or correction"}})
 with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False,indent=2)
 print(f"done: elapsed={time.time()-start:.1f}s, output={out.resolve()}")

if __name__=="__main__": main()
