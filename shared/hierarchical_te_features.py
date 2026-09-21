#!/usr/bin/env python3
"""공식 train 과거 시즌만 사용하는 계층형 상황 Target Encoding."""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET="control_success"
SPECS={
 "p_game":(["pitcher_id","game_type"],"pitcher_id",45.),
 "p_count":(["pitcher_id","_count"],"pitcher_id",55.),
 "p_bhand":(["pitcher_id","batter_hand"],"pitcher_id",55.),
 "p_inning":(["pitcher_id","_inning"],"pitcher_id",70.),
 "p_pressure":(["pitcher_id","_pressure"],"pitcher_id",70.),
 "b_game":(["batter_id","game_type"],"batter_id",65.),
 "b_count":(["batter_id","_count"],"batter_id",75.),
 "b_phand":(["batter_id","pitcher_hand"],"batter_id",65.),
 "pt_game":(["pitcher_team_id","game_type"],"pitcher_team_id",700.),
 "pt_count":(["pitcher_team_id","_count"],"pitcher_team_id",900.),
 "bt_game":(["batter_team_id","game_type"],"batter_team_id",700.),
}
PARENT_PRIOR={"pitcher_id":80.,"batter_id":80.,"pitcher_team_id":800.,"batter_team_id":800.}

def context(df):
 z=df.copy(); balls=pd.to_numeric(z["balls_before"],errors="coerce").fillna(0).astype(int); strikes=pd.to_numeric(z["strikes_before"],errors="coerce").fillna(0).astype(int)
 z["_count"]=(balls*4+strikes).astype(str); z["_inning"]=pd.to_numeric(z["inning"],errors="coerce").fillna(1).clip(1,10).astype(int).astype(str)
 li=pd.to_numeric(z["li"],errors="coerce").fillna(1); runners=pd.to_numeric(z["num_runners_on"],errors="coerce").fillna(0); z["_pressure"]=np.select([(li>=2)|(balls>=3), (li>=1.2)|(runners>=2)],["high","mid"],default="low")
 return z

def _key_frame(df,cols):
 out=df[cols[0]].astype(str)
 for c in cols[1:]: out=out+"|"+df[c].astype(str)
 return out

def fit_artifact(history):
 h=context(history); gm=float(h[TARGET].mean()) if len(h) else .5; parent={}; parent_count={}
 for col,prior in PARENT_PRIOR.items():
  s=h.groupby(col,observed=True)[TARGET].agg(["sum","count"]); parent[col]={str(k):float((r["sum"]+prior*gm)/(r["count"]+prior)) for k,r in s.iterrows()}; parent_count[col]={str(k):int(r["count"]) for k,r in s.iterrows()}
 combos={}
 for name,(cols,pcol,prior) in SPECS.items():
  if not len(h): combos[name]={}; continue
  q=pd.DataFrame({"_key":_key_frame(h,cols),TARGET:h[TARGET],pcol:h[pcol]}); s=q.groupby("_key",observed=True).agg(sum=(TARGET,"sum"),count=(TARGET,"count"),parent=(pcol,"first")); mp={}
  for key,r in s.iterrows():
   pv=parent[pcol].get(str(r["parent"]),gm); mp[str(key)]={"v":float((r["sum"]+prior*pv)/(r["count"]+prior)),"n":int(r["count"])}
  combos[name]=mp
 return {"global_mean":gm,"parent":parent,"parent_count":parent_count,"combos":combos}

def transform(df,art):
 z=context(df); out=pd.DataFrame(index=df.index); gm=float(art["global_mean"])
 for name,(cols,pcol,_) in SPECS.items():
  parent=z[pcol].astype(str).map(art["parent"][pcol]).fillna(gm).astype(np.float32); key=_key_frame(z,cols); mp=art["combos"][name]
  out[f"hte_{name}"]=key.map({k:v["v"] for k,v in mp.items()}).fillna(parent).astype(np.float32)
  out[f"hte_{name}_logn"]=np.log1p(key.map({k:v["n"] for k,v in mp.items()}).fillna(0)).astype(np.float32)
  out[f"hte_{name}_delta"]=(out[f"hte_{name}"]-parent).astype(np.float32)
 return out

def temporal_transform(df):
 parts=[]
 for season in sorted(df.season.unique()):
  rows=df[df.season==season]; part=transform(rows,fit_artifact(df[df.season<season])); part.index=rows.index; parts.append(part)
 return pd.concat(parts).sort_index()
