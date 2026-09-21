#!/usr/bin/env python3
"""2023/2024 walk-forward evaluation of FactorTabM against frozen DSF OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pipelines.dsf_upgrade.factor_tabm import FactorTabM, Preprocessor, make_features, predict
from pipelines.trackman.command_features import build_command_artifacts, enrich_command

ID, TARGET = "row_id", "control_success"


def brier(y, p): return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))
def bss(y, p):
    y = np.asarray(y, float); return 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean())))


def fit_model(cats, nums, y, weights, vc, vn, vy, pre, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(6)
    model = FactorTabM(pre.card, nums.shape[1], pre.pair_indices, k=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=4e-4)
    best, best_loss = None, float("inf"); batch = 16384
    for epoch in range(epochs):
        model.train(); order = np.random.default_rng(seed + epoch).permutation(len(y))
        for start in range(0, len(order), batch):
            ix = order[start:start + batch]
            logits = model(torch.as_tensor(cats[ix]), torch.as_tensor(nums[ix])); target = torch.as_tensor(y[ix])[:, None]
            probability = torch.sigmoid(logits); member = nn.functional.binary_cross_entropy_with_logits(logits, target.expand_as(logits), reduction="none")
            loss = ((0.55 * member.mean(dim=1) + 0.20 * (probability - target).square().mean(dim=1) + 0.25 * (probability.mean(dim=1) - target[:, 0]).square()) * torch.as_tensor(weights[ix])).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 4.0); optimizer.step()
        pv = predict(model, vc, vn); score = brier(vy, pv)
        print(f"epoch={epoch+1} stop_brier={score:.8f}", flush=True)
        if score < best_loss:
            best_loss = score; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best); return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--dsf_oof", required=True); ap.add_argument("--out_dir", default="./factor_tabm_oof")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--trackman_path", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False); frame[ID] = frame[ID].astype(str)
    x = make_features(frame)
    if args.trackman_path:
        command_artifact = build_command_artifacts(args.trackman_path)
        command = enrich_command(frame, command_artifact).reset_index(drop=True)
        x = pd.concat([x.reset_index(drop=True), command], axis=1)
    y = frame[TARGET].to_numpy(np.float32); seasons = frame.season.to_numpy(int)
    with np.load(args.dsf_oof, allow_pickle=False) as saved:
        dsf = pd.DataFrame({ID: saved["row_id"].astype(str), "dsf": saved["blend_061"].astype(float), "saved_y": saved["y"].astype(float)})
    results = {}
    for year in (2023, 2024):
        train_mask, valid_mask = seasons < year, seasons == year
        hashes = pd.util.hash_pandas_object(frame.loc[valid_mask, ID], index=False).to_numpy(); stop = (hashes % 4) < 2; select = (hashes % 4) == 2; evaluate = (hashes % 4) == 3
        pre = Preprocessor().fit(x.loc[train_mask]); ct, nt = pre.transform(x.loc[train_mask]); cv, nv = pre.transform(x.loc[valid_mask])
        train_years = seasons[train_mask]; weights = (1.0 + 0.10 * (train_years - train_years.min())).astype(np.float32); weights /= weights.mean()
        model = fit_model(ct, nt, y[train_mask], weights, cv[stop], nv[stop], y[valid_mask][stop], pre, 44000 + year + args.seed_offset, args.epochs)
        pred = predict(model, cv, nv); ids = frame.loc[valid_mask, ID].to_numpy(); local = pd.DataFrame({ID: ids, "y": y[valid_mask], "factor": pred, "stop": stop, "select": select, "evaluate": evaluate}).merge(dsf, on=ID, validate="one_to_one")
        if not np.array_equal(local.y.to_numpy(float), local.saved_y.to_numpy(float)): raise RuntimeError("target mismatch")
        matrix = local[["dsf", "factor"]].to_numpy(float); target = local.y.to_numpy(float); weights_grid = np.arange(0.0, 0.201, 0.01)
        select_local = local["select"].to_numpy(bool); evaluate_local = local["evaluate"].to_numpy(bool)
        chosen = min(weights_grid, key=lambda w: brier(target[select_local], (1-w)*matrix[select_local,0]+w*matrix[select_local,1]))
        blend = (1-chosen)*matrix[:,0]+chosen*matrix[:,1]
        corr = float(np.corrcoef(matrix[evaluate_local]-target[evaluate_local,None], rowvar=False)[0,1])
        result = {"weight": float(chosen), "dsf_bss": bss(target[evaluate_local], matrix[evaluate_local,0]), "factor_bss": bss(target[evaluate_local], matrix[evaluate_local,1]), "blend_bss": bss(target[evaluate_local], blend[evaluate_local]), "residual_correlation": corr, "n_eval": int(evaluate_local.sum())}
        results[str(year)] = result
        np.savez_compressed(out / f"factor_{year}.npz", row_id=local[ID].to_numpy(dtype=str), y=target.astype(np.float32), dsf=matrix[:,0].astype(np.float32), factor=matrix[:,1].astype(np.float32), stop=local["stop"].to_numpy(bool), select=select_local, evaluate=evaluate_local)
        print(year, json.dumps(result), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
