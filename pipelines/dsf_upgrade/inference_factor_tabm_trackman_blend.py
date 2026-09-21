#!/usr/bin/env python3
"""DSF + champion FactorTabM router + paper-inspired TrackMan FactorTabM."""
from __future__ import annotations

import importlib.util,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from pipelines.dsf_upgrade.factor_tabm import FactorTabM,make_features,predict
from pipelines.trackman.command_features import enrich_command

DATA_DIR,MODEL_DIR,OUT_PATH="./data","./model","./output/submission.csv";ID,TARGET="row_id","control_success"

def _load_predict(path,pre,features):
    artifact=torch.load(path,map_location="cpu",weights_only=True);cats,nums=pre.transform(features);model=FactorTabM(artifact["cardinalities"],artifact["n_num"],artifact["pair_indices"],k=artifact["k"]);model.load_state_dict(artifact["state_dict"]);return predict(model,cats,nums)

def _router_features(dsf,factor):
    diff=factor-dsf;return np.column_stack([dsf,factor,diff,np.abs(diff),np.abs(dsf-.5),np.abs(factor-.5),dsf*(1-dsf)])

def main():
    root=Path(__file__).resolve().parent;out=Path(OUT_PATH);out.parent.mkdir(parents=True,exist_ok=True);test=pd.read_csv(Path(DATA_DIR)/"test.csv",encoding="utf-8-sig",low_memory=False);sample=pd.read_csv(Path(DATA_DIR)/"sample_submission.csv",encoding="utf-8-sig")
    dsf_root=root/"dsf";spec=importlib.util.spec_from_file_location("_trackman_blend_dsf",dsf_root/"member_script.py");module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(dsf_root))
    try:spec.loader.exec_module(module);module.DATA_DIR=DATA_DIR;module.MODEL_DIR=str(dsf_root/"model");module.OUT_PATH=str(out.parent/"_dsf.csv");module.main()
    finally:sys.path.pop(0)
    raw=pd.read_csv(out.parent/"_dsf.csv");dsf=pd.Series(raw[TARGET].to_numpy(float),index=raw[ID].astype(str)).reindex(test[ID].astype(str)).to_numpy();model_dir=Path(MODEL_DIR);base_features=make_features(test)
    base=_load_predict(model_dir/"base_factor.pt",joblib.load(model_dir/"base_preprocessor.joblib"),base_features);router=joblib.load(model_dir/"router.joblib");route=router.predict_proba(_router_features(dsf,base))[:,1]
    meta=json.loads((model_dir/"metadata.json").read_text());champion=dsf+(route>=meta["router_threshold"])*meta["router_alpha"]*(base-dsf)
    command=json.loads((model_dir/"trackman_command_artifacts.json").read_text());track_features=pd.concat([base_features.reset_index(drop=True),enrich_command(test,command).reset_index(drop=True)],axis=1)
    track_factor=_load_predict(model_dir/"track_factor.pt",joblib.load(model_dir/"track_preprocessor.joblib"),track_features);track_candidate=(1-meta["track_factor_weight"])*dsf+meta["track_factor_weight"]*track_factor
    prediction=(1-meta["track_candidate_share"])*champion+meta["track_candidate_share"]*track_candidate;aligned=pd.Series(np.clip(prediction,0,1),index=test[ID].astype(str)).reindex(sample[ID].astype(str)).to_numpy()
    if not np.isfinite(aligned).all():raise RuntimeError("row alignment failed")
    sample[TARGET]=aligned;sample.to_csv(out,index=False);print(f"DSF/FactorTabM/TrackMan rows={len(sample)} routed={int((route>=meta['router_threshold']).sum())}")


if __name__=="__main__":main()
