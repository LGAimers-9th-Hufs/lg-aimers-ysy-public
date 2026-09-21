#!/usr/bin/env python3
"""Train final FactorTabM and the frozen 2023-OOF conditional router."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from pipelines.dsf_upgrade.factor_tabm import FactorTabM, Preprocessor, make_features
from pipelines.trackman.command_features import build_command_artifacts, enrich_command, save_command_artifacts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train_path", default="./data/train.csv")
    ap.add_argument("--factor_oof_2023", default="./factor_tabm_oof/factor_2023.npz")
    ap.add_argument("--output_dir", default="./factor_tabm_final"); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--min_season", type=int, default=2020)
    ap.add_argument("--seed", type=int, default=59025)
    ap.add_argument("--trackman_path", default=None)
    args = ap.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    frame = frame.loc[frame["season"].to_numpy(int) >= args.min_season].reset_index(drop=True)
    features = make_features(frame)
    if args.trackman_path:
        command_artifact=build_command_artifacts(args.trackman_path)
        features=pd.concat([features.reset_index(drop=True),enrich_command(frame,command_artifact).reset_index(drop=True)],axis=1)
        save_command_artifacts(command_artifact,output/"trackman_command_artifacts.json")
    target = frame["control_success"].to_numpy(np.float32)
    pre = Preprocessor().fit(features); cats, nums = pre.transform(features); n_num = nums.shape[1]
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    model = FactorTabM(pre.card, n_num, pre.pair_indices, k=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=4e-4); batch = 8192
    years = frame["season"].to_numpy(int); weights = (1 + .10 * (years-years.min())).astype(np.float32); weights /= weights.mean()
    del frame, years; gc.collect()
    for epoch in range(args.epochs):
        model.train(); order = np.random.default_rng(args.seed + epoch).permutation(len(target)); running = 0.0
        for start in range(0, len(order), batch):
            ix = order[start:start+batch]
            c = torch.as_tensor(cats[ix]); n = torch.as_tensor(nums[ix]); y = torch.as_tensor(target[ix])[:,None]
            logits = model(c, n); probability = torch.sigmoid(logits)
            member = nn.functional.binary_cross_entropy_with_logits(logits, y.expand_as(logits), reduction="none")
            loss = ((.55*member.mean(1) + .20*(probability-y).square().mean(1) + .25*(probability.mean(1)-y[:,0]).square()) * torch.as_tensor(weights[ix])).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 4.0); optimizer.step(); running += float(loss)*len(ix)
            del c, n, y, logits, probability, member, loss
        print(f"epoch={epoch+1} loss={running/len(target):.8f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "cardinalities": pre.card, "n_num": n_num, "pair_indices": pre.pair_indices, "k": 8}, output / "factor_tabm.pt")
    joblib.dump(pre, output / "preprocessor.joblib", compress=3)

    (output / "metadata.json").write_text(json.dumps({"threshold": .5, "alpha": .3, "epochs": args.epochs, "min_season": args.min_season, "seed": args.seed, "trackman_command":bool(args.trackman_path),
        "router_fit": "2023 walk-forward OOF stop hash only", "rules": "official train only; row-local inference"}, indent=2), encoding="utf-8")
    print(f"saved {output.resolve()}")


if __name__ == "__main__": main()
