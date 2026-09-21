#!/usr/bin/env python3
"""저장된 네 OOF 배열만 이용하는 경량 결합 계산."""
import json
import numpy as np
from scipy.optimize import minimize
from pipelines.catboost.experiments.evaluate_catboost_blend import fit_cal,calibrate,bss

def aligned(z,rid):
 order={v:i for i,v in enumerate(z["row_id"].astype(str))}
 idx=np.asarray([order[v] for v in rid])
 return z["pred"][idx].astype(np.float64),idx

c=np.load("/private/tmp/catboost_base_2024_oof.npz",allow_pickle=True)
m=np.load("/private/tmp/mlp_walkforward/mlp_2024_oof.npz",allow_pickle=True)
t=np.load("/private/tmp/trackman_2024_oof.npz",allow_pickle=True)
rid=c["row_id"].astype(str); pm,_=aligned(m,rid); pt,idx=aligned(t,rid)
y=t["y"][idx]; cf=t["cal_fit"][idx]; ev=t["eval"][idx]
a=np.column_stack([c["full"],c["recent"],pm,pt]); names=["full_catboost","recent3_catboost","embedding_mlp","trackman_physics"]
def obj(w): return float(np.mean((a[cf]@w-y[cf])**2))
opts=[]
for s in (np.full(4,.25),np.array([.387748,.273565,.338687,0]),np.array([0,0,0,1])):
 r=minimize(obj,s,method="SLSQP",bounds=[(0,1)]*4,constraints={"type":"eq","fun":lambda w:w.sum()-1},options={"maxiter":1000,"ftol":1e-15})
 w=np.clip(r.x,0,1); w/=w.sum(); opts.append((obj(w),w))
w=min(opts,key=lambda q:q[0])[1]; q=a@w; cal=fit_cal(y[cf],q[cf])
out={"weights":dict(zip(names,w.tolist())),"calibration":cal,"blend_raw_eval":bss(y[ev],q[ev]),"blend_cal_eval":bss(y[ev],calibrate(q[ev],cal)),"individual_eval":{n:bss(y[ev],a[ev,i]) for i,n in enumerate(names)},"residual_correlation_eval":np.corrcoef(a[ev]-y[ev,None],rowvar=False).tolist(),"rules":"official train OOF + official TrackMan only; no test or leaderboard information"}
with open("/private/tmp/trackman_ensemble.json","w") as f: json.dump(out,f,indent=2)
np.savez_compressed("/private/tmp/ensemble_components_2024.npz",row_id=rid,y=y,full=a[:,0],recent=a[:,1],mlp=a[:,2],trackman=a[:,3],blend=q,cal_fit=cf,eval=ev)
print(json.dumps(out,indent=2))
