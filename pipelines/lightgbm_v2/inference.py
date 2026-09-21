#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""규정 준수 추론: test 각 행과 train-only 정적 artifact만 사용한다."""
from __future__ import annotations
import json
import os
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COL, TARGET_COL = "row_id", "control_success"
DATA_DIR, MODEL_DIR = Path("./data"), Path("./model")
OUT_PATH = Path("./output/submission.csv")

def _num(df, col, default=np.nan):
    if col not in df:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").astype(np.float32)

def build_features(df, art):
    """행 단위 산술과 train-only 정적 lookup만 수행한다."""
    missing = [c for c in art["required_input_cols"] if c not in df]
    if missing:
        raise ValueError(f"입력 컬럼 누락: {missing}")
    x = pd.DataFrame(index=df.index)
    for c in art["raw_numeric_cols"]:
        x[c] = _num(df, c)
    for c, mapping in art["cat_maps"].items():
        x[f"cat_{c}"] = df[c].astype(str).map(mapping).fillna(-1).astype(np.int16)

    gm = float(art["global_mean"])
    p_n = _num(df, "asof_pitcher_n", 0).fillna(0).clip(lower=0)
    b_n = _num(df, "asof_batter_n", 0).fillna(0).clip(lower=0)
    mix_n = _num(df, "asof_pitcher_pitchmix_n", 0).fillna(0).clip(lower=0)
    p_rate = _num(df, "asof_pitcher_success_rate").fillna(gm)
    b_rate = _num(df, "asof_batter_success_rate").fillna(gm)
    x["p_logn"], x["b_logn"] = np.log1p(p_n).astype(np.float32), np.log1p(b_n).astype(np.float32)
    x["mix_logn"] = np.log1p(mix_n).astype(np.float32)
    for prior in (25.0, 100.0, 400.0):
        tag = int(prior)
        x[f"p_sm{tag}"] = ((p_rate*p_n + prior*gm)/(p_n+prior)).astype(np.float32)
        x[f"b_sm{tag}"] = ((b_rate*b_n + prior*gm)/(b_n+prior)).astype(np.float32)

    balls, strikes = _num(df, "balls_before", 0).fillna(0), _num(df, "strikes_before", 0).fillna(0)
    inning, runners = _num(df, "inning", 1).fillna(1), _num(df, "num_runners_on", 0).fillna(0)
    li, score = _num(df, "li", 1).fillna(1), _num(df, "score_diff_pitcher_team", 0).fillna(0)
    x["count_code"] = (balls*4+strikes).astype(np.int8)
    x["two_strike"], x["three_ball"] = (strikes>=2).astype(np.int8), (balls>=3).astype(np.int8)
    x["full_count"] = ((balls==3)&(strikes==2)).astype(np.int8)
    x["zero_zero"] = ((balls==0)&(strikes==0)).astype(np.int8)
    x["risp"] = ((_num(df,"runner_on_2b",0).fillna(0)>0)|(_num(df,"runner_on_3b",0).fillna(0)>0)).astype(np.int8)
    x["is_close"], x["late_inning"], x["high_li"] = (score.abs()<=2).astype(np.int8), (inning>=7).astype(np.int8), (li>=2).astype(np.int8)
    x["pressure"] = (li*(1+x["three_ball"])*(1+runners)).astype(np.float32)
    x["li_x_runners"], x["count_pressure"] = (li*runners).astype(np.float32), ((balls-strikes)*li).astype(np.float32)
    x["hand_match"] = (df["pitcher_hand"].astype(str)==df["batter_hand"].astype(str)).astype(np.int8)

    fb = _num(df,"asof_pitcher_fastball_rate").fillna(.5).clip(lower=0)
    br = _num(df,"asof_pitcher_breaking_rate").fillna(.25).clip(lower=0)
    off = _num(df,"asof_pitcher_offspeed_rate").fillna(.25).clip(lower=0)
    total = (fb+br+off).replace(0,1.0); fb,br,off = fb/total,br/total,off/total
    x["mix_fb"],x["mix_br"],x["mix_off"] = fb.astype(np.float32),br.astype(np.float32),off.astype(np.float32)
    x["mix_entropy"] = (-(fb*np.log(fb+1e-7)+br*np.log(br+1e-7)+off*np.log(off+1e-7))).astype(np.float32)

    for c in art["te_cols"]:
        x[f"te_{c}"] = df[c].astype(str).map(art["te_maps"].get(c,{})).fillna(gm).astype(np.float32)
    x["te_player_diff"] = (x["te_pitcher_id"]-x["te_batter_id"]).astype(np.float32)
    x["te_player_prod"] = (x["te_pitcher_id"]*x["te_batter_id"]).astype(np.float32)
    x["p_skill_diff"] = (x["p_sm100"]-x["b_sm100"]).astype(np.float32)
    x["p_skill_pressure"] = (x["p_sm100"]*x["pressure"]).astype(np.float32)
    x["p_skill_two_strike"] = (x["p_sm100"]*x["two_strike"]).astype(np.float32)
    x["p_middle_pressure"] = (_num(df,"asof_pitcher_middle_rate").fillna(.3)*x["pressure"]).astype(np.float32)

    expected = art.get("feature_cols")
    if expected is None:
        return x
    miss, extra = [c for c in expected if c not in x], [c for c in x if c not in expected]
    if miss or extra:
        raise ValueError(f"피처 스키마 불일치 missing={miss}, extra={extra}")
    return x[expected]

def apply_calibration(pred, cal):
    p=np.clip(np.asarray(pred,dtype=np.float64),1e-6,1-1e-6)
    z=np.clip(float(cal["a"])*np.log(p/(1-p))+float(cal["b"]),-30,30)
    return 1/(1+np.exp(-z))

def load_bundle(model_dir=MODEL_DIR):
    with open(model_dir/"artifacts.json",encoding="utf-8") as f: art=json.load(f)
    models=[lgb.Booster(model_file=str(model_dir/f)) for f in art["model_files"]]
    if len(models)!=len(art["blend_weights"]): raise ValueError("모델/weight 수 불일치")
    return models,art

def predict_frame(df,models,art):
    x=build_features(df,art); w=np.asarray(art["blend_weights"],dtype=np.float64); w/=w.sum()
    pred=np.zeros(len(df),dtype=np.float64)
    for model,weight in zip(models,w):
        if model.feature_name()!=art["feature_cols"]: raise ValueError("모델/artifact 피처 불일치")
        pred += weight*model.predict(x,num_threads=int(art.get("predict_threads",6)))
    return apply_calibration(pred,art["calibration"])

def main():
    models,art=load_bundle()
    test=pd.read_csv(DATA_DIR/"test.csv",encoding="utf-8-sig")
    sub=pd.read_csv(DATA_DIR/"sample_submission.csv",encoding="utf-8-sig")
    if test[ID_COL].isna().any() or not test[ID_COL].is_unique: raise ValueError("test row_id 결측/중복")
    if sub[ID_COL].isna().any() or not sub[ID_COL].is_unique: raise ValueError("submission row_id 결측/중복")
    if len(test)!=len(sub) or set(test[ID_COL].astype(str))!=set(sub[ID_COL].astype(str)):
        raise ValueError("test/submission row_id 불일치")
    pred=predict_frame(test,models,art)
    if not np.isfinite(pred).all(): raise ValueError("NaN/Inf 예측")
    pred_map=dict(zip(test[ID_COL].astype(str),pred))
    sub[TARGET_COL]=sub[ID_COL].astype(str).map(pred_map)
    if sub[TARGET_COL].isna().any(): raise ValueError("누락 예측")
    os.makedirs(OUT_PATH.parent,exist_ok=True); sub.to_csv(OUT_PATH,index=False,encoding="utf-8")
    # Do not compute even diagnostic distribution statistics over test outputs.
    print(f"saved {OUT_PATH}: rows={len(sub)}")

if __name__=="__main__": main()
