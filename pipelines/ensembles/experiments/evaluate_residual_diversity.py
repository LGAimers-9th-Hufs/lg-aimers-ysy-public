#!/usr/bin/env python3
"""CatBoost full/recent + PyTorch MLP의 OOF residual-diversity 탐색."""
from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from shared.catboost_features import ALL_CAT_COLS,ID_COL,TARGET_COL,build_features
from pipelines.catboost.experiments.evaluate_catboost_blend import fit_oof,full_params,recent_params,fit_cal,calibrate,bss

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--mlp_oof",required=True); ap.add_argument("--out",default="./residual_diversity.json"); a=ap.parse_args(); st=time.time()
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=build_features(df); y=df[TARGET_COL].to_numpy(np.int8); va=df["season"].to_numpy()==2024; xv,yv=x.loc[va],y[va]
 cats=[c for c in ALL_CAT_COLS if c in x]; ci=[x.columns.get_loc(c) for c in cats]; h=pd.util.hash_pandas_object(df.loc[va,ID_COL].astype(str),index=False).to_numpy(); stop=(h%4)<2; cf=(h%4)==2; ev=(h%4)==3
 full=df["season"].to_numpy()<2024; fy=df.loc[full,"season"].to_numpy(); fw=(1+.18*(fy-2019)).astype(np.float32)
 pf,itf=fit_oof("full",x,y,df,full,xv,yv,stop,ci,fw,full_params)
 recent=df["season"].isin([2021,2022,2023]).to_numpy(); ry=df.loc[recent,"season"].to_numpy(); rw=(1+.15*(ry-2021)).astype(np.float32)
 pr,itr=fit_oof("recent3",x,y,df,recent,xv,yv,stop,ci,rw,recent_params)
 z=np.load(a.mlp_oof,allow_pickle=True); rid=df.loc[va,ID_COL].astype(str).to_numpy(); order={v:i for i,v in enumerate(z["row_id"].astype(str))}; pm=np.asarray([z["pred"][order[v]] for v in rid],dtype=np.float64)
 matrix=np.column_stack([pf,pr,pm]); names=["full_catboost","recent3_catboost","embedding_mlp"]
 def obj(w): return float(np.mean((matrix[cf]@w-yv[cf])**2))
 opt=minimize(obj,np.full(3,1/3),method="SLSQP",bounds=[(0.,1.)]*3,constraints={"type":"eq","fun":lambda w:w.sum()-1.},options={"maxiter":500,"ftol":1e-14}); w=np.clip(opt.x,0,1); w/=w.sum(); blend=matrix@w; cal=fit_cal(yv[cf],blend[cf])
 residuals=matrix[ev]-yv[ev,None]; corr=np.corrcoef(residuals,rowvar=False)
 # Explicit residual-correlation grid: select lowest calibration Brier among candidates
 # whose residual correlation with the current 2-way blend is reduced by >= 0.01.
 base=.6059713556780735*pf+.3940286443219265*pr; base_res=base[cf]-yv[cf]; grid=[]
 for mw in np.linspace(0,.5,26):
  for fwgt in np.linspace(0,1-mw,31):
   rwgt=1-mw-fwgt; q=fwgt*pf+rwgt*pr+mw*pm; rc=float(np.corrcoef(q[cf]-yv[cf],base_res)[0,1]); bs=float(np.mean((q[cf]-yv[cf])**2)); grid.append((bs,rc,fwgt,rwgt,mw))
 eligible=[g for g in grid if g[1]<=.99]; diverse=min(eligible,key=lambda g:g[0]) if eligible else min(grid,key=lambda g:g[1]); dw=np.array(diverse[2:]); dq=matrix@dw; dc=fit_cal(yv[cf],dq[cf])
 out={"best_iterations":{"full":itf,"recent3":itr},"brier_opt_weights":dict(zip(names,w.tolist())),"brier_opt_calibration":cal,"brier_opt_eval":bss(yv[ev],calibrate(blend[ev],cal)),"individual_eval":{n:bss(yv[ev],matrix[ev,i]) for i,n in enumerate(names)},"residual_correlation_eval":corr.tolist(),"diversity_candidate":{"weights":dict(zip(names,dw.tolist())),"cal_residual_corr_with_current":diverse[1],"eval":bss(yv[ev],calibrate(dq[ev],dc)),"calibration":dc},"elapsed_seconds":time.time()-st,"rules":"official train OOF only; no test or leaderboard information"}
 with open(a.out,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
