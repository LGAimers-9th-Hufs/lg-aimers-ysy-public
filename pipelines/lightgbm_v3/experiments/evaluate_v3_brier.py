#!/usr/bin/env python3
"""동일 V3 피처의 binary classifier와 직접 Brier(L2) regressor 비교."""
from __future__ import annotations
import argparse,json,time
import lightgbm as lgb
import numpy as np
import pandas as pd
from pipelines.lightgbm_v3.train import bss,fit_calibration,calibrate,temporal_features

def params(seed):
 return dict(objective="regression_l2",metric="l2",boosting_type="gbdt",n_estimators=1200,num_leaves=63,max_depth=8,min_child_samples=260,learning_rate=.018,feature_fraction=.88,bagging_fraction=.9,bagging_freq=1,reg_alpha=.3,reg_lambda=3.5,verbosity=-1,n_jobs=6,seed=seed,deterministic=True,force_col_wise=True)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--base_oof",required=True); ap.add_argument("--out",default="./v3_brier.json"); ap.add_argument("--pred_out",default="./v3_brier_oof.npz"); a=ap.parse_args(); st=time.time()
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); y=df.control_success.to_numpy(np.float32); x=temporal_features(df); base=np.load(a.base_oof,allow_pickle=True); border={v:i for i,v in enumerate(base["row_id"].astype(str))}; results={}; saved=[]
 for yr in (2022,2023,2024):
  tr=df.season.to_numpy()<yr; va=df.season.to_numpy()==yr; rid=df.loc[va,"row_id"].astype(str).to_numpy(); bix=np.asarray([border[v] for v in rid]); pv=base["v3"][bix]; h=pd.util.hash_pandas_object(df.loc[va,"row_id"].astype(str),index=False).to_numpy(); stop=(h%4)<2; cf=(h%4)==2; ev=(h%4)==3; w=(1+.18*(df.loc[tr,"season"].to_numpy()-2019)).astype(np.float32)
  m=lgb.LGBMRegressor(**params(9300+yr)); m.fit(x.loc[tr],y[tr],sample_weight=w,eval_set=[(x.loc[va].loc[stop],y[va][stop])],callbacks=[lgb.early_stopping(140,verbose=False),lgb.log_evaluation(200)]); p=np.clip(m.predict(x.loc[va]),0,1); cal=fit_calibration(y[va][cf],p[cf]); results[str(yr)]={"v3_raw_eval":bss(y[va][ev],pv[ev]),"brier_iterations":int(m.best_iteration_),"brier_raw_eval":bss(y[va][ev],p[ev]),"calibration":cal,"brier_cal_eval":bss(y[va][ev],calibrate(p[ev],cal)),"prediction_corr":float(np.corrcoef(pv[ev],p[ev])[0,1])}; saved.append((rid,y[va],pv,p,np.full(va.sum(),yr),cf,ev)); print(yr,results[str(yr)],flush=True)
 arrays=[np.concatenate([s[i] for s in saved]) for i in range(7)]; np.savez_compressed(a.pred_out,row_id=arrays[0],y=arrays[1],v3=arrays[2],brier=arrays[3],season=arrays[4],cal_fit=arrays[5],eval=arrays[6]); out={"folds":results,"elapsed_seconds":time.time()-st,"rules":"official train only; past-season OOF; no test or leaderboard information"}
 with open(a.out,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
