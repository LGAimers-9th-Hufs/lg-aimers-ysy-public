#!/usr/bin/env python3
"""V3와 계층형 상황 TE V3의 2022--2024 walk-forward 비교."""
from __future__ import annotations
import argparse,json,time
import lightgbm as lgb
import numpy as np
import pandas as pd
from pipelines.lightgbm_v2.inference import build_features
from pipelines.lightgbm_v3.train import base_artifact,bss,fit_calibration,calibrate,temporal_features
from shared.hierarchical_te_features import temporal_transform

def params(seed,rounds):
 return dict(objective="binary",metric="binary_logloss",boosting_type="gbdt",n_estimators=rounds,num_leaves=95,max_depth=10,min_child_samples=220,learning_rate=.02,feature_fraction=.86,bagging_fraction=.88,bagging_freq=1,reg_alpha=.25,reg_lambda=2.5,verbosity=-1,n_jobs=6,seed=seed,deterministic=True,force_col_wise=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--out",default="./hierarchical_v3.json"); ap.add_argument("--pred_out",default="./hierarchical_v3_oof.npz"); a=ap.parse_args(); st=time.time()
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); y=df.control_success.to_numpy(np.float32); print("building temporal V3 features",flush=True); xb=temporal_features(df); print("building hierarchical TE",flush=True); xh=pd.concat([xb,temporal_transform(df)],axis=1)
 results={}; saved=[]
 for yr in (2022,2023,2024):
  tr=df.season.to_numpy()<yr; va=df.season.to_numpy()==yr; rid=df.loc[va,"row_id"].astype(str).to_numpy(); h=pd.util.hash_pandas_object(df.loc[va,"row_id"].astype(str),index=False).to_numpy(); stop=(h%4)<2; cf=(h%4)==2; ev=(h%4)==3; w=(1+.18*(df.loc[tr,"season"].to_numpy()-2019)).astype(np.float32); fold={}
  preds=[]
  for name,x in (("v3",xb),("hierarchical",xh)):
   m=lgb.LGBMClassifier(**params(6100+yr+(31 if name=="hierarchical" else 0),1000)); m.fit(x.loc[tr],y[tr],sample_weight=w,eval_set=[(x.loc[va].loc[stop],y[va][stop])],callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(200)]); p=m.predict_proba(x.loc[va])[:,1]; cal=fit_calibration(y[va][cf],p[cf]); fold[name]={"iterations":int(m.best_iteration_),"raw_eval":bss(y[va][ev],p[ev]),"calibration":cal,"cal_eval":bss(y[va][ev],calibrate(p[ev],cal))}; preds.append(p); print(yr,name,fold[name],flush=True)
  results[str(yr)]=fold; saved.append((rid,y[va],preds[0],preds[1],np.full(va.sum(),yr),cf,ev))
 arrays=[np.concatenate([s[i] for s in saved]) for i in range(7)]; np.savez_compressed(a.pred_out,row_id=arrays[0],y=arrays[1],v3=arrays[2],hierarchical=arrays[3],season=arrays[4],cal_fit=arrays[5],eval=arrays[6])
 out={"folds":results,"elapsed_seconds":time.time()-st,"rules":"official train only; every TE fold uses strictly earlier seasons; no test or leaderboard information"};
 with open(a.out,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
