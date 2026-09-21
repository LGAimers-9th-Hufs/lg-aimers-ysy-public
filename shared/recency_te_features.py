#!/usr/bin/env python3
"""V3용 train-only 선수/팀 최근성 Bayesian Target Encoding."""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET="control_success"
COLS=("pitcher_id","batter_id","pitcher_team_id","batter_team_id")
PRIORS={"pitcher_id":80.,"batter_id":80.,"pitcher_team_id":800.,"batter_team_id":800.}

def _plain_map(h,col,gm,prior):
 s=h.groupby(col,observed=True)[TARGET].agg(["sum","count"])
 return ({str(k):float((r["sum"]+prior*gm)/(r["count"]+prior)) for k,r in s.iterrows()},
         {str(k):float(r["count"]) for k,r in s.iterrows()})

def _weighted_map(h,col,gm,prior,cutoff,decay=.62):
 if not len(h): return {},{}
 w=np.power(decay,(cutoff-1-h["season"].to_numpy()).clip(min=0)); q=pd.DataFrame({"key":h[col].astype(str),"wy":w*h[TARGET].to_numpy(),"w":w}); s=q.groupby("key",observed=True)[["wy","w"]].sum()
 return ({str(k):float((r["wy"]+prior*gm)/(r["w"]+prior)) for k,r in s.iterrows()},
         {str(k):float(r["w"]) for k,r in s.iterrows()})

def fit_artifact(history,cutoff):
 gm=float(history[TARGET].mean()) if len(history) else .5; art={"global_mean":gm,"columns":{}}
 for col in COLS:
  prior=PRIORS[col]; full,full_n=_plain_map(history,col,gm,prior); rec={"full":full,"full_n":full_n}
  for years in (1,2,3):
   z=history[history.season>=cutoff-years]; mp,n=_plain_map(z,col,gm,prior); rec[f"last{years}"]=mp; rec[f"last{years}_n"]=n
  exp,exp_n=_weighted_map(history,col,gm,prior,cutoff); rec["exp"]=exp; rec["exp_n"]=exp_n; art["columns"][col]=rec
 return art

def transform(df,art):
 out=pd.DataFrame(index=df.index); gm=float(art["global_mean"])
 for col in COLS:
  key=df[col].astype(str); prior=PRIORS[col]; a=art["columns"][col]; full=key.map(a["full"]).fillna(gm).astype(np.float32); n=key.map(a["full_n"]).fillna(0).astype(np.float32)
  out[f"rte_{col}_logn"]=np.log1p(n).astype(np.float32); out[f"rte_{col}_reliability"]=(n/(n+prior)).astype(np.float32)
  for tag in ("last1","last2","last3","exp"):
   v=key.map(a[tag]).fillna(full).astype(np.float32); rn=key.map(a[tag+"_n"]).fillna(0).astype(np.float32)
   out[f"rte_{col}_{tag}"]=v; out[f"rte_{col}_{tag}_logn"]=np.log1p(rn).astype(np.float32); out[f"rte_{col}_{tag}_delta"]=(v-full).astype(np.float32)
  out[f"rte_{col}_trend_1_3"]=(out[f"rte_{col}_last1"]-out[f"rte_{col}_last3"]).astype(np.float32)
 return out

def temporal_transform(df):
 parts=[]
 for season in sorted(df.season.unique()):
  rows=df[df.season==season]; part=transform(rows,fit_artifact(df[df.season<season],int(season))); part.index=rows.index; parts.append(part)
 return pd.concat(parts).sort_index()
