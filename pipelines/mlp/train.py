#!/usr/bin/env python3
"""PyTorch embedding MLP 최종 2019~2024 학습."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from pipelines.mlp.train_walkforward import Preprocessor,SharedTrunkMLP,enrich
from shared.catboost_features import TARGET_COL

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--out_dir",default="./model_mlp"); ap.add_argument("--epochs",type=int,default=2); a=ap.parse_args()
 out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); st=time.time(); torch.manual_seed(15026); np.random.seed(15026); torch.set_num_threads(6)
 df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=enrich(df); y=df[TARGET_COL].to_numpy(np.float32); pre=Preprocessor().fit(x); cats,num=pre.transform(x)
 years=df["season"].to_numpy(); weights=(1+.12*(years-years.min())).astype(np.float32); weights/=weights.mean(); device=torch.device("cpu")
 model=SharedTrunkMLP(pre.card,num.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1.8e-3,weight_decay=2e-4); batch=8192
 for epoch in range(a.epochs):
  model.train(); order=np.random.default_rng(15026+epoch).permutation(len(y)); total=0.
  for s in range(0,len(y),batch):
   ix=order[s:s+batch]; c=torch.as_tensor(cats[ix]); n=torch.as_tensor(num[ix]); t=torch.as_tensor(y[ix]); w=torch.as_tensor(weights[ix])
   logit=model(c,n); p=torch.sigmoid(logit); loss=((.72*nn.functional.binary_cross_entropy_with_logits(logit,t,reduction="none")+.28*(p-t).square())*w).mean()
   opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss)*len(ix)
  print(f"epoch={epoch+1} loss={total/len(y):.6f}",flush=True)
 torch.save({"state_dict":model.state_dict(),"cardinalities":pre.card,"n_num":num.shape[1]},out/"mlp.pt")
 with open(out/"preprocess.json","w",encoding="utf-8") as f: json.dump(pre.artifact(),f,ensure_ascii=False)
 art={"schema_version":1,"pipeline":"pytorch_embedding_shared_trunk","epochs":a.epochs,"seed":15026,"feature_cols":x.columns.tolist(),"cat_cols":pre.cat_cols,"num_cols":pre.num_cols,"architecture":{"hidden":[256,128,64],"dropout":[.12,.08],"loss":"0.72 BCE + 0.28 Brier"},"train_rows":len(df),"train_seasons":sorted(df.season.unique().tolist()),"rules":"official train only; row-independent inference; no test aggregates"}
 with open(out/"artifacts.json","w",encoding="utf-8") as f: json.dump(art,f,ensure_ascii=False,indent=2)
 print(f"done {time.time()-st:.1f}s -> {out}")
if __name__=="__main__": main()
