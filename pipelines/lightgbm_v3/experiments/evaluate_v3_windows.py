#!/usr/bin/env python3
"""동일 V3 피처에서 전체 과거, 최근 3시즌, 최근 2시즌 학습 범위 비교."""
from __future__ import annotations
import argparse,json,time
import lightgbm as lgb
import numpy as np
import pandas as pd
from pipelines.lightgbm_v3.train import bss,fit_calibration,calibrate,temporal_features

def params(seed):
 return dict(objective="binary",metric="binary_logloss",boosting_type="gbdt",n_estimators=1000,num_leaves=95,max_depth=10,min_child_samples=220,learning_rate=.02,feature_fraction=.86,bagging_fraction=.88,bagging_freq=1,reg_alpha=.25,reg_lambda=2.5,verbosity=-1,n_jobs=6,seed=seed,deterministic=True,force_col_wise=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--base_oof",required=True); ap.add_argument("--out",default="./v3_windows.json"); ap.add_argument("--pred_out",default="./v3_windows_oof.npz"); a=ap.parse_args(); st=time.time()
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); y=df.control_success.to_numpy(np.float32); x=temporal_features(df); base=np.load(a.base_oof,allow_pickle=True); border={v:i for i,v in enumerate(base["row_id"].astype(str))}; results={}; saved=[]
 for yr in (2022,2023,2024):
  va=df.season.to_numpy()==yr; rid=df.loc[va,"row_id"].astype(str).to_numpy(); bix=np.asarray([border[v] for v in rid]); pfull=base["v3"][bix]; h=pd.util.hash_pandas_object(df.loc[va,"row_id"].astype(str),index=False).to_numpy(); stop=(h%4)<2; cf=(h%4)==2; ev=(h%4)==3; fold={"full":{"raw_eval":bss(y[va][ev],pfull[ev])}}; preds=[pfull]
  for j,years in enumerate((3,2)):
   tr=(df.season.to_numpy()<yr)&(df.season.to_numpy()>=yr-years); sy=df.loc[tr,"season"].to_numpy(); w=(1+.18*(sy-sy.min())).astype(np.float32); m=lgb.LGBMClassifier(**params(8400+yr+j*79)); m.fit(x.loc[tr],y[tr],sample_weight=w,eval_set=[(x.loc[va].loc[stop],y[va][stop])],callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(200)]); p=m.predict_proba(x.loc[va])[:,1]; cal=fit_calibration(y[va][cf],p[cf]); name=f"recent{years}"; fold[name]={"train_seasons":sorted(df.loc[tr,"season"].unique().tolist()),"iterations":int(m.best_iteration_),"raw_eval":bss(y[va][ev],p[ev]),"calibration":cal,"cal_eval":bss(y[va][ev],calibrate(p[ev],cal))}; preds.append(p); print(yr,name,fold[name],flush=True)
  results[str(yr)]=fold; saved.append((rid,y[va],*preds,np.full(va.sum(),yr),cf,ev))
 arrays=[np.concatenate([s[i] for s in saved]) for i in range(8)]; np.savez_compressed(a.pred_out,row_id=arrays[0],y=arrays[1],full=arrays[2],recent3=arrays[3],recent2=arrays[4],season=arrays[5],cal_fit=arrays[6],eval=arrays[7])
 out={"folds":results,"elapsed_seconds":time.time()-st,"rules":"official train only; every model trains only seasons before validation; no test or leaderboard information"}
 with open(a.out,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
