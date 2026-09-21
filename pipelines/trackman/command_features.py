#!/usr/bin/env python3
"""Paper-inspired, train-frozen TrackMan command-consistency priors.

Anonymous competition pitcher IDs cannot be linked to TrackMan pitcher IDs.  We
therefore estimate pitcher-season consistency inside official TrackMan history,
then aggregate those profiles by prior season, throwing hand, and pitch group.
Serving only looks up the frozen prior and combines it with row-local pitch mix.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS=("fastball","breaking","offspeed")
RAW=("rel_speed","spin_rate","induced_vert_break","horz_break","extension","rel_height","rel_side","zone_speed")
PROFILE_METRICS=(
    "release_side_std","release_height_std","release_ellipse","release_major","release_minor","release_corr",
    "speed_std","spin_std","extension_std","movement_ellipse","movement_major","movement_minor","movement_corr",
    "active_spin_proxy_mean","active_spin_proxy_std","velo_retention_std",
    "drift_release","drift_speed","drift_movement",
)


def _profiles(path: str | Path, min_pitches: int=20) -> pd.DataFrame:
    use=["season","pitcher_trackman_id","pitcher_hand","pitch_type_group",*RAW]
    z=pd.read_csv(path,usecols=use,encoding="utf-8-sig",low_memory=False)
    z=z.loc[z["pitch_type_group"].isin(GROUPS)].copy()
    for column in RAW:z[column]=pd.to_numeric(z[column],errors="coerce")
    z["pitcher_hand"]=z["pitcher_hand"].map({"Right":"R","Left":"L"}).fillna("U")
    z["release_product"]=z["rel_side"]*z["rel_height"]
    z["movement_product"]=z["horz_break"]*z["induced_vert_break"]
    z["active_spin_proxy"]=np.hypot(z["induced_vert_break"],z["horz_break"])*1000.0/z["spin_rate"].replace(0,np.nan)
    z["velo_retention"]=z["zone_speed"]/z["rel_speed"].replace(0,np.nan)
    keys=["season","pitcher_trackman_id","pitcher_hand","pitch_type_group"]
    agg=z.groupby(keys,observed=True,sort=False).agg(
        n=("rel_speed","count"),speed_mean=("rel_speed","mean"),speed_std=("rel_speed","std"),spin_std=("spin_rate","std"),
        extension_std=("extension","std"),side_mean=("rel_side","mean"),release_side_std=("rel_side","std"),
        height_mean=("rel_height","mean"),release_height_std=("rel_height","std"),release_product=("release_product","mean"),
        ivb_mean=("induced_vert_break","mean"),ivb_std=("induced_vert_break","std"),hb_mean=("horz_break","mean"),hb_std=("horz_break","std"),
        movement_product=("movement_product","mean"),active_spin_proxy_mean=("active_spin_proxy","mean"),active_spin_proxy_std=("active_spin_proxy","std"),
        velo_retention_std=("velo_retention","std"),
    ).reset_index()
    agg=agg.loc[agg["n"]>=min_pitches].copy()
    release_cov=agg["release_product"]-agg["side_mean"]*agg["height_mean"]
    movement_cov=agg["movement_product"]-agg["hb_mean"]*agg["ivb_mean"]
    def ellipse(std_x,std_y,cov,prefix):
        vx=std_x.fillna(0).to_numpy(float)**2;vy=std_y.fillna(0).to_numpy(float)**2;c=cov.fillna(0).to_numpy(float)
        trace=vx+vy;disc=np.sqrt(np.maximum((vx-vy)**2+4*c*c,0));major=np.sqrt(np.maximum((trace+disc)/2,0));minor=np.sqrt(np.maximum((trace-disc)/2,0))
        agg[prefix+"_ellipse"]=(np.pi*5.991*major*minor).astype(np.float32);agg[prefix+"_major"]=major.astype(np.float32);agg[prefix+"_minor"]=minor.astype(np.float32)
        agg[prefix+"_corr"]=(c/np.maximum(np.sqrt(vx*vy),1e-8)).clip(-1,1).astype(np.float32)
    ellipse(agg["release_side_std"],agg["release_height_std"],release_cov,"release")
    ellipse(agg["hb_std"],agg["ivb_std"],movement_cov,"movement")
    agg=agg.sort_values(["pitcher_trackman_id","pitch_type_group","season"])
    previous=agg.groupby(["pitcher_trackman_id","pitch_type_group"],observed=True,sort=False)[["side_mean","height_mean","speed_mean","hb_mean","ivb_mean"]].shift(1)
    agg["drift_release"]=np.hypot(agg["side_mean"]-previous["side_mean"],agg["height_mean"]-previous["height_mean"])
    agg["drift_speed"]=(agg["speed_mean"]-previous["speed_mean"]).abs()
    agg["drift_movement"]=np.hypot(agg["hb_mean"]-previous["hb_mean"],agg["ivb_mean"]-previous["ivb_mean"])
    return agg[[*keys,"n",*PROFILE_METRICS]]


def _weighted_record(frame: pd.DataFrame, cutoff: int, hand: str | None, group: str) -> dict[str,float]:
    q=frame.loc[(frame["season"]<cutoff)&frame["pitch_type_group"].eq(group)]
    if hand is not None:q=q.loc[q["pitcher_hand"].eq(hand)]
    if not len(q):return {metric:0.0 for metric in PROFILE_METRICS}|{"n_profiles":0}
    recency=np.power(.72,(cutoff-1-q["season"].to_numpy(int)).clip(min=0));reliability=np.sqrt(np.minimum(q["n"].to_numpy(float),400)/400);weight=recency*reliability
    result={"n_profiles":int(len(q))}
    for metric in PROFILE_METRICS:
        values=q[metric].to_numpy(float);valid=np.isfinite(values)
        result[metric]=float(np.average(values[valid],weights=weight[valid])) if valid.any() else 0.0
    return result


def build_command_artifacts(path: str | Path) -> dict:
    profiles=_profiles(path);cutoffs={}
    for cutoff in range(2019,2026):
        hand_group={};group={}
        for pitch_group in GROUPS:
            group[pitch_group]=_weighted_record(profiles,cutoff,None,pitch_group)
            for hand in ("L","R"):hand_group[f"{hand}|{pitch_group}"]=_weighted_record(profiles,cutoff,hand,pitch_group)
        cutoffs[str(cutoff)]={"hand_group":hand_group,"group":group}
    return {"schema_version":1,"groups":list(GROUPS),"metrics":list(PROFILE_METRICS),"cutoffs":cutoffs,
        "rules":"official TrackMan only; pitcher-season profiles use seasons strictly before prediction season; no ID linkage"}


def save_command_artifacts(artifact: dict,path: str | Path):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(artifact,ensure_ascii=False),encoding="utf-8")


def enrich_command(df: pd.DataFrame,artifact: dict) -> pd.DataFrame:
    season=pd.to_numeric(df["season"],errors="coerce").fillna(2025).astype(int).clip(2019,2025).to_numpy();hand=df["pitcher_hand"].astype(str).map({"1":"L","2":"R","L":"L","R":"R"}).fillna("U").to_numpy()
    mix=np.column_stack([pd.to_numeric(df[f"asof_pitcher_{g}_rate"],errors="coerce").to_numpy(float) for g in GROUPS]);mix=np.nan_to_num(mix,nan=1/3);mix=np.clip(mix,0,None);den=mix.sum(1);mix[den<=0]=1/3;mix/=mix.sum(1)[:,None]
    values={metric:np.zeros((len(df),3),np.float32) for metric in PROFILE_METRICS};counts=np.zeros((len(df),3),np.float32)
    for year in np.unique(season):
        for throwing_hand in np.unique(hand[season==year]):
            rows=(season==year)&(hand==throwing_hand);profile=artifact["cutoffs"][str(int(year))]
            for j,group in enumerate(GROUPS):
                record=profile["hand_group"].get(f"{throwing_hand}|{group}") or profile["group"][group];counts[rows,j]=np.log1p(record["n_profiles"])
                for metric in PROFILE_METRICS:values[metric][rows,j]=record[metric]
    out=pd.DataFrame(index=df.index)
    for metric in PROFILE_METRICS:
        out[f"tmcmd_{metric}"]=(values[metric]*mix).sum(1).astype(np.float32)
        out[f"tmcmd_fb_br_{metric}_gap"]=(values[metric][:,0]-values[metric][:,1]).astype(np.float32)
    out["tmcmd_profile_logn"]=(counts*mix).sum(1).astype(np.float32)
    out["tmcmd_release_risk"]=(out["tmcmd_release_ellipse"]*(1+out["tmcmd_drift_release"])).astype(np.float32)
    out["tmcmd_movement_risk"]=(out["tmcmd_movement_ellipse"]*(1+out["tmcmd_drift_movement"])).astype(np.float32)
    return out
