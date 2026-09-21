#!/usr/bin/env python3
"""규정 준수 PyTorch embedding MLP walk-forward OOF."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from shared.catboost_features import build_features,TARGET_COL,ID_COL

BASE_CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand","pitcher_id","batter_id","pitcher_team_id","batter_team_id","count_state","runner_state","team_matchup","hand_matchup","inning_bucket"]
EXTRA_CAT=["regime","game_type_regime","game_type_pitcher_team","game_type_batter_team","game_type_count"]

def bss(y,p):
 y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64); bs=float(np.mean((p-y)**2)); ref=float(y.mean()*(1-y.mean()))
 return max(0.,100000*(1-bs/ref)),bs

def enrich(df):
 x=build_features(df)
 season=pd.to_numeric(df["season"],errors="coerce").fillna(0).astype(int); regime=np.where(season>=2023,"post23","pre23")
 gt=df["game_type"].fillna("__MISSING__").astype(str); pt=df["pitcher_team_id"].fillna("__MISSING__").astype(str); bt=df["batter_team_id"].fillna("__MISSING__").astype(str)
 x["regime"]=regime; x["game_type_regime"]=gt+"_"+pd.Series(regime,index=df.index).astype(str)
 x["game_type_pitcher_team"]=gt+"_"+pt; x["game_type_batter_team"]=gt+"_"+bt; x["game_type_count"]=gt+"_"+x["count_state"].astype(str)
 return x

class Preprocessor:
 def fit(self,x):
  self.cat_cols=[c for c in BASE_CAT+EXTRA_CAT if c in x]; self.num_cols=[c for c in x if c not in self.cat_cols]
  self.maps={}; self.card=[]
  for c in self.cat_cols:
   vals=pd.Series(x[c].fillna("__MISSING__").astype(str).unique()).sort_values().tolist(); self.maps[c]={v:i+1 for i,v in enumerate(vals)}; self.card.append(len(vals)+1)
  num=x[self.num_cols].apply(pd.to_numeric,errors="coerce"); self.median=num.median().fillna(0.); filled=num.fillna(self.median)
  self.mean=filled.mean(); self.std=filled.std().replace(0,1).fillna(1.); return self
 def transform(self,x):
  cats=np.column_stack([x[c].fillna("__MISSING__").astype(str).map(self.maps[c]).fillna(0).to_numpy(np.int64) for c in self.cat_cols])
  num=x[self.num_cols].apply(pd.to_numeric,errors="coerce").fillna(self.median); num=((num-self.mean)/self.std).clip(-8,8).to_numpy(np.float32)
  return cats,num
 def artifact(self):
  return {"cat_cols":self.cat_cols,"num_cols":self.num_cols,"maps":self.maps,"cardinalities":self.card,"median":self.median.to_dict(),"mean":self.mean.to_dict(),"std":self.std.to_dict()}

class SharedTrunkMLP(nn.Module):
 def __init__(self,cardinalities,n_num):
  super().__init__(); dims=[min(24,max(4,int(round(1.6*(c**.25))))) for c in cardinalities]
  self.emb=nn.ModuleList([nn.Embedding(c,d) for c,d in zip(cardinalities,dims)])
  inp=sum(dims)+n_num
  self.trunk=nn.Sequential(nn.Linear(inp,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.12),nn.Linear(256,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.08))
  self.head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
 def forward(self,cat,num):
  z=torch.cat([e(cat[:,i]) for i,e in enumerate(self.emb)]+[num],dim=1); return self.head(self.trunk(z)).squeeze(1)

def predict(model,cats,num,device,batch=16384):
 model.eval(); out=[]
 with torch.no_grad():
  for s in range(0,len(num),batch):
   c=torch.as_tensor(cats[s:s+batch],device=device); n=torch.as_tensor(num[s:s+batch],device=device)
   out.append(torch.sigmoid(model(c,n)).cpu().numpy())
 return np.concatenate(out)

def train_fold(c_train,n_train,y_train,w_train,c_stop,n_stop,y_stop,card,device,seed,max_epochs):
 torch.manual_seed(seed); np.random.seed(seed); model=SharedTrunkMLP(card,n_train.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1.8e-3,weight_decay=2e-4)
 best=None; best_bs=1e9; patience=0; batch=8192
 for epoch in range(max_epochs):
  model.train(); order=np.random.default_rng(seed+epoch).permutation(len(y_train)); total=0.
  for s in range(0,len(order),batch):
   ix=order[s:s+batch]; c=torch.as_tensor(c_train[ix],device=device); n=torch.as_tensor(n_train[ix],device=device); y=torch.as_tensor(y_train[ix],device=device); w=torch.as_tensor(w_train[ix],device=device)
   logit=model(c,n); prob=torch.sigmoid(logit); loss=((.72*nn.functional.binary_cross_entropy_with_logits(logit,y,reduction="none")+.28*(prob-y).square())*w).mean()
   opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss)*len(ix)
  ps=predict(model,c_stop,n_stop,device); bs=float(np.mean((ps-y_stop)**2)); print(f" epoch={epoch+1} loss={total/len(y_train):.6f} stop_brier={bs:.8f}",flush=True)
  if bs<best_bs-2e-6: best_bs=bs; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
  else: patience+=1
  if patience>=3: break
 if best is not None: model.load_state_dict(best)
 return model,best_bs,epoch+1

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--train_path",default="./data/train.csv"); ap.add_argument("--out_dir",default="./mlp_walkforward"); ap.add_argument("--folds",default="2022,2023,2024"); ap.add_argument("--epochs",type=int,default=16)
 a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); st=time.time(); df=pd.read_csv(a.train_path,encoding="utf-8-sig"); x=enrich(df); y=df[TARGET_COL].to_numpy(np.float32)
 device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"); print("device",device,flush=True); results={}; pred_files={}
 for year in [int(v) for v in a.folds.split(",") if v]:
  tr=df["season"].to_numpy()<year; va=df["season"].to_numpy()==year; xv=x.loc[va]; yv=y[va]
  h=pd.util.hash_pandas_object(df.loc[va,ID_COL].astype(str),index=False).to_numpy(); stop=(h%4)<2; ev=(h%4)==3
  pre=Preprocessor().fit(x.loc[tr]); ct,nt=pre.transform(x.loc[tr]); cv,nv=pre.transform(xv)
  years=df.loc[tr,"season"].to_numpy(); w=(1+.12*(years-years.min())).astype(np.float32); w/=w.mean()
  print(f"[{year}] train={tr.sum():,} val={va.sum():,} cats={ct.shape[1]} nums={nt.shape[1]}",flush=True)
  model,stop_bs,epochs=train_fold(ct,nt,y[tr],w,cv[stop],nv[stop],yv[stop],pre.card,device,11000+year,a.epochs)
  pv=predict(model,cv,nv,device); score=bss(yv[ev],pv[ev]); name=f"mlp_{year}"; torch.save({"state_dict":model.state_dict(),"cardinalities":pre.card,"n_num":nt.shape[1]},out/f"{name}.pt")
  with open(out/f"{name}_preprocess.json","w",encoding="utf-8") as f: json.dump(pre.artifact(),f,ensure_ascii=False)
  np.savez_compressed(out/f"{name}_oof.npz",row_id=df.loc[va,ID_COL].astype(str).to_numpy(),y=yv,pred=pv,stop=stop,eval=ev)
  results[str(year)]={"raw_bss":score[0],"raw_brier":score[1],"stop_brier":stop_bs,"epochs":epochs,"n_train":int(tr.sum()),"n_val":int(va.sum())}; pred_files[str(year)]=f"{name}_oof.npz"
  print(f"[{year}] untouched BSS={score[0]:.3f}",flush=True)
 results["summary"]={"mean_bss":float(np.mean([v["raw_bss"] for v in results.values()])),"std_bss":float(np.std([v["raw_bss"] for v in results.values()])),"worst_bss":float(np.min([v["raw_bss"] for v in results.values()]))}
 results["device"]=str(device); results["elapsed_seconds"]=time.time()-st; results["rules"]="fold preprocess fit on strictly earlier official train seasons; row-wise inference only"
 with open(out/"results.json","w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
 print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
