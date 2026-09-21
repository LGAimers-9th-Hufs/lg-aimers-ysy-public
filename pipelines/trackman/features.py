#!/usr/bin/env python3
"""공식 과거 TrackMan 로그의 시점 안전 계층 프로파일과 행 독립 피처."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS=("fastball","breaking","offspeed")
RAW_METRICS=("rel_speed","spin_rate","induced_vert_break","horz_break","extension","rel_height","rel_side","zone_speed")
METRICS=("rel_speed","spin_rate","induced_vert_break","abs_horz_break","extension","rel_height","abs_rel_side","zone_speed","velo_retention")

def _prepare(tm):
    z=tm.loc[tm["pitch_type_group"].isin(GROUPS),["season","pitcher_hand","pitch_type_group",*RAW_METRICS]].copy()
    for c in RAW_METRICS: z[c]=pd.to_numeric(z[c],errors="coerce")
    z["abs_horz_break"]=z["horz_break"].abs(); z["abs_rel_side"]=z["rel_side"].abs()
    z["velo_retention"]=z["zone_speed"]/z["rel_speed"].replace(0,np.nan)
    z["pitcher_hand"]=z["pitcher_hand"].map({"Right":"R","Left":"L"}).fillna("U")
    return z[["season","pitcher_hand","pitch_type_group",*METRICS]]

def _weighted_stats(z,cutoff,keys,decay=.72):
    q=z[z.season<cutoff].copy(); q["w"]=np.power(decay,(cutoff-1-q.season).clip(lower=0))
    out={}
    for key,g in q.groupby(keys,observed=True,sort=False):
        if not isinstance(key,tuple): key=(key,)
        rec={"n":int(len(g))}
        w=g["w"].to_numpy(np.float64)
        for m in METRICS:
            v=g[m].to_numpy(np.float64); ok=np.isfinite(v)
            if not ok.any(): rec[m+"_mean"]=None; rec[m+"_std"]=None; continue
            ww=w[ok]; vv=v[ok]; mu=float(np.average(vv,weights=ww)); var=float(np.average((vv-mu)**2,weights=ww))
            rec[m+"_mean"]=mu; rec[m+"_std"]=float(np.sqrt(max(var,0)))
        out["|".join(map(str,key))]=rec
    return out

def build_artifacts(trackman_path):
    tm=pd.read_csv(trackman_path,encoding="utf-8-sig"); z=_prepare(tm); cutoffs={}
    for cutoff in range(2019,2026):
        cutoffs[str(cutoff)]={
            "hand_group":_weighted_stats(z,cutoff,["pitcher_hand","pitch_type_group"]),
            "group":_weighted_stats(z,cutoff,["pitch_type_group"]),
            "global":_weighted_stats(z,cutoff,[]).get("",{}) if False else {},
        }
        # Explicit global record avoids pandas groupby([]).
        q=z[z.season<cutoff].copy(); q["_all"]="all"
        cutoffs[str(cutoff)]["global"]=_weighted_stats(q,cutoff,["_all"]).get("all",{})
    return {"schema_version":1,"cutoffs":cutoffs,"metrics":list(METRICS),"groups":list(GROUPS),
            "rules":"official trackman_history only; each row uses profiles from seasons strictly before row season"}

def save_artifacts(art,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False)

def load_artifacts(path):
    with open(path,encoding="utf-8") as f: return json.load(f)

def _record(profile,hand,group):
    return profile["hand_group"].get(f"{hand}|{group}") or profile["group"].get(group) or profile["global"]

def enrich_trackman(df,art):
    """현재 df 행의 값만으로 고정 TrackMan artifact를 lookup한다."""
    n=len(df); season=pd.to_numeric(df["season"],errors="coerce").fillna(2025).astype(int).clip(2019,2025)
    hand=df["pitcher_hand"].astype(str).map({"1":"L","2":"R","L":"L","R":"R"}).fillna("U")
    rates={
        "fastball":pd.to_numeric(df["asof_pitcher_fastball_rate"],errors="coerce"),
        "breaking":pd.to_numeric(df["asof_pitcher_breaking_rate"],errors="coerce"),
        "offspeed":pd.to_numeric(df["asof_pitcher_offspeed_rate"],errors="coerce"),
    }
    mix=np.column_stack([rates[g].to_numpy(np.float64) for g in GROUPS]); mix=np.nan_to_num(mix,nan=1/3,posinf=0,neginf=0); mix=np.clip(mix,0,None)
    den=mix.sum(axis=1); mix[den<=0]=1/3; den=mix.sum(axis=1); mix/=den[:,None]
    means={m:np.empty((n,3),np.float32) for m in METRICS}; stds={m:np.empty((n,3),np.float32) for m in METRICS}; counts=np.empty((n,3),np.float32)
    for i,(yr,h) in enumerate(zip(season,hand)):
        profile=art["cutoffs"][str(int(yr))]
        for j,g in enumerate(GROUPS):
            rec=_record(profile,h,g); counts[i,j]=np.log1p(float(rec.get("n",0)))
            for m in METRICS:
                means[m][i,j]=float(rec.get(m+"_mean") or 0.0); stds[m][i,j]=float(rec.get(m+"_std") or 0.0)
    out=pd.DataFrame(index=df.index)
    for m in METRICS:
        out[f"tm_exp_{m}"]=(means[m]*mix).sum(axis=1).astype(np.float32)
        out[f"tm_exp_{m}_std"]=(stds[m]*mix).sum(axis=1).astype(np.float32)
        out[f"tm_fb_br_{m}_gap"]=(means[m][:,0]-means[m][:,1]).astype(np.float32)
    out["tm_profile_logn"]=(counts*mix).sum(axis=1).astype(np.float32)
    out["tm_mix_entropy"]=(-(mix*np.log(mix+1e-8)).sum(axis=1)).astype(np.float32)
    out["tm_movement_separation"]=(np.sqrt((means["induced_vert_break"][:,0]-means["induced_vert_break"][:,1])**2+(means["abs_horz_break"][:,0]-means["abs_horz_break"][:,1])**2)).astype(np.float32)
    return out
