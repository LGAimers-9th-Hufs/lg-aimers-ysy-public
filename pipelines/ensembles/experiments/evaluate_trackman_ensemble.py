#!/usr/bin/env python3
"""기존 3모델 + TrackMan 물리 모델의 strict OOF Brier 앙상블."""
from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from shared.catboost_features import ALL_CAT_COLS,ID_COL,TARGET_COL,build_features
from pipelines.catboost.experiments.evaluate_catboost_blend import fit_oof,full_params,recent_params,fit_cal,calibrate,bss

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--mlp_oof",required=True); ap.add_argument("--trackman_oof",required=True); ap.add_argument("--out",default="./trackman_ensemble.json"); ap.add_argument("--pred_out",default="./ensemble_components_2024.npz"); ap.add_argument("--cat_cache",default="/private/tmp/catboost_base_2024_oof.npz"); a=ap.parse_args(); st=time.time()
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=build_features(df); y=df[TARGET_COL].to_numpy(np.int8); va=df.season.to_numpy()==2024; xv=x.loc[va]; yv=y[va]
 cats=[c for c in ALL_CAT_COLS if c in x]; ci=[x.columns.get_loc(c) for c in cats]; h=pd.util.hash_pandas_object(df.loc[va,ID_COL].astype(str),index=False).to_numpy(); stop=(h%4)<2; cf=(h%4)==2; ev=(h%4)==3
 rid=df.loc[va,ID_COL].astype(str).to_numpy()
 try:
  cache=np.load(a.cat_cache,allow_pickle=True); same=np.array_equal(cache["row_id"].astype(str),rid)
 except (FileNotFoundError,KeyError,ValueError): same=False
 if same:
  pf=cache["full"].astype(np.float64); pr=cache["recent"].astype(np.float64); itf=int(cache["it_full"]); itr=int(cache["it_recent"]); print("[cache] loaded CatBoost OOF")
 else:
  full=df.season.to_numpy()<2024; fy=df.loc[full,"season"].to_numpy(); fw=(1+.18*(fy-2019)).astype(np.float32); pf,itf=fit_oof("full",x,y,df,full,xv,yv,stop,ci,fw,full_params)
  recent=df.season.isin([2021,2022,2023]).to_numpy(); ry=df.loc[recent,"season"].to_numpy(); rw=(1+.15*(ry-2021)).astype(np.float32); pr,itr=fit_oof("recent3",x,y,df,recent,xv,yv,stop,ci,rw,recent_params)
  np.savez_compressed(a.cat_cache,row_id=rid,full=pf,recent=pr,it_full=itf,it_recent=itr)
 def aligned(path):
  z=np.load(path,allow_pickle=True); order={v:i for i,v in enumerate(z["row_id"].astype(str))}; return np.asarray([z["pred"][order[v]] for v in rid],dtype=np.float64)
 pm=aligned(a.mlp_oof); pt=aligned(a.trackman_oof); matrix=np.column_stack([pf,pr,pm,pt]); names=["full_catboost","recent3_catboost","embedding_mlp","trackman_physics"]
 def obj(w): return float(np.mean((matrix[cf]@w-yv[cf])**2))
 starts=[np.full(4,.25),np.array([.387748,.273565,.338687,0]),np.array([0,0,0,1])]; opts=[]
 for s in starts:
  r=minimize(obj,s,method="SLSQP",bounds=[(0.,1.)]*4,constraints={"type":"eq","fun":lambda w:w.sum()-1},options={"maxiter":1000,"ftol":1e-15}); w=np.clip(r.x,0,1); w/=w.sum(); opts.append((obj(w),w))
 _,w=min(opts,key=lambda q:q[0]); blend=matrix@w; cal=fit_cal(yv[cf],blend[cf]); pc=calibrate(blend,cal); res=matrix[ev]-yv[ev,None]
 out={"best_iterations":{"full":itf,"recent3":itr},"weights":dict(zip(names,w.tolist())),"calibration":cal,"blend_raw_eval":bss(yv[ev],blend[ev]),"blend_cal_eval":bss(yv[ev],pc[ev]),"individual_eval":{n:bss(yv[ev],matrix[ev,i]) for i,n in enumerate(names)},"residual_correlation_eval":np.corrcoef(res,rowvar=False).tolist(),"elapsed_seconds":time.time()-st,"rules":"official train OOF + official historical TrackMan only; no test or leaderboard information"}
 with open(a.out,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
 np.savez_compressed(a.pred_out,row_id=rid,y=yv,full=pf,recent=pr,mlp=pm,trackman=pt,blend=blend,stop=stop,cal_fit=cf,eval=ev)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
