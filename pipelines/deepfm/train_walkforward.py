#!/usr/bin/env python3
"""Field-aware DeepFM trained on official prior-season rows only."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
TARGET = "control_success"
BASE_CATEGORICAL = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "pitcher_id", "batter_id", "num_runners_on",
]
NUMERIC = [
    "li", "home_win_expectancy", "score_diff_pitcher_team",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def categorical(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[BASE_CATEGORICAL].astype("string").fillna("<NA>").copy()
    count = x["balls_before"] + "-" + x["strikes_before"]
    hand = x["pitcher_hand"] + "-" + x["batter_hand"]
    x["count"] = count
    x["count_hand"] = count + "|" + hand
    x["count_base"] = count + "|" + x["base_state"]
    x["regime_context"] = x["game_type"] + "|" + count + "|" + x["base_state"]
    x["pitcher_count"] = x["pitcher_id"] + "|" + count
    x["pitcher_bhand"] = x["pitcher_id"] + "|" + x["batter_hand"]
    x["batter_count"] = x["batter_id"] + "|" + count
    x["team_matchup"] = x["pitcher_team_id"] + "|" + x["batter_team_id"]
    x["pitcher_regime"] = x["pitcher_id"] + "|" + x["game_type"]
    return x


def encode(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    a = categorical(train)
    b = categorical(valid)
    train_columns = []
    valid_columns = []
    sizes = []
    mappings = {}
    for column in a.columns:
        values = pd.Index(a[column].unique())
        mapping = {str(value): index + 1 for index, value in enumerate(values)}
        train_columns.append(a[column].astype(str).map(mapping).fillna(0).to_numpy(np.int64))
        valid_columns.append(b[column].astype(str).map(mapping).fillna(0).to_numpy(np.int64))
        sizes.append(len(mapping) + 1)
        mappings[column] = mapping
    return np.column_stack(train_columns), np.column_stack(valid_columns), sizes, mappings


def numeric(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    a = train[NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    b = valid[NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    for column in ("asof_pitcher_n", "asof_batter_n"):
        index = NUMERIC.index(column)
        a[:, index] = np.log1p(np.maximum(a[:, index], 0.0))
        b[:, index] = np.log1p(np.maximum(b[:, index], 0.0))
    median = np.nanmedian(a, axis=0)
    median[~np.isfinite(median)] = 0.0
    for values in (a, b):
        missing = np.where(~np.isfinite(values))
        values[missing] = median[missing[1]]
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std[std < 1e-5] = 1.0
    return (a - mean) / std, (b - mean) / std, mean.tolist(), std.tolist()


class DeepFM(nn.Module):
    def __init__(self, sizes: list[int], numeric_count: int, dimension: int = 8) -> None:
        super().__init__()
        self.linear = nn.ModuleList([nn.Embedding(size, 1) for size in sizes])
        self.embedding = nn.ModuleList([nn.Embedding(size, dimension) for size in sizes])
        width = len(sizes) * dimension + numeric_count
        self.deep = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Dropout(0.10), nn.Linear(128, 48), nn.SiLU(), nn.Linear(48, 1))
        self.numeric_linear = nn.Linear(numeric_count, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        for table in [*self.linear, *self.embedding]:
            nn.init.normal_(table.weight, std=0.01)

    def forward(self, categorical_values: torch.Tensor, numeric_values: torch.Tensor) -> torch.Tensor:
        linear = torch.stack([table(categorical_values[:, i]).squeeze(-1) for i, table in enumerate(self.linear)], dim=1).sum(dim=1)
        embeddings = torch.stack([table(categorical_values[:, i]) for i, table in enumerate(self.embedding)], dim=1)
        summed = embeddings.sum(dim=1)
        fm = 0.5 * (summed.square() - embeddings.square().sum(dim=1)).sum(dim=1)
        deep = self.deep(torch.cat([embeddings.flatten(1), numeric_values], dim=1)).squeeze(-1)
        return self.bias + linear + self.numeric_linear(numeric_values).squeeze(-1) + fm + deep


def bss(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    yy = y[mask]
    return float(100000.0 * (1.0 - np.mean((yy - prediction[mask]) ** 2) / (yy.mean() * (1.0 - yy.mean()))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/deepfm_2024"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    usecols = list(dict.fromkeys(["row_id", "season", TARGET, *BASE_CATEGORICAL, *NUMERIC]))
    frame = pd.read_csv(args.train, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    train = frame.loc[frame["season"].eq(2023)].reset_index(drop=True)
    valid = frame.loc[frame["season"].eq(2024)].reset_index(drop=True)
    train_cat, valid_cat, sizes, mappings = encode(train, valid)
    train_num, valid_num, means, stds = numeric(train, valid)
    y_train = train[TARGET].to_numpy(np.float32)
    y_valid = valid[TARGET].to_numpy(np.float32)
    dataset = TensorDataset(torch.from_numpy(train_cat), torch.from_numpy(train_num), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = DeepFM(sizes, len(NUMERIC))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for cat_batch, num_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = torch.sigmoid(model(cat_batch, num_batch))
            loss = torch.mean((prediction - target_batch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(target_batch)
        print(f"epoch={epoch + 1} train_brier={total / len(dataset):.8f}", flush=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(valid), args.batch_size):
            cat_batch = torch.from_numpy(valid_cat[start:start + args.batch_size])
            num_batch = torch.from_numpy(valid_num[start:start + args.batch_size])
            predictions.append(torch.sigmoid(model(cat_batch, num_batch)).numpy())
    prediction = np.concatenate(predictions).astype(np.float64)
    mask_file = np.load("/private/tmp/regime_stack_2024_oof.npz", allow_pickle=True)
    if not np.array_equal(valid["row_id"].astype(str).to_numpy(), mask_file["row_id"].astype(str)):
        raise RuntimeError("validation row alignment failed")
    report = {
        "standalone": {"select": bss(y_valid, prediction, mask_file["select"]), "evaluate": bss(y_valid, prediction, mask_file["evaluate"])},
        "rows": {"train": len(train), "valid": len(valid)}, "epochs": args.epochs,
        "rules": "2023 official train only -> 2024 OOF; row-local fields; Brier loss; no external/test aggregation/LB feedback",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "sizes": sizes, "mappings": mappings, "means": means, "stds": stds, "numeric": NUMERIC}, args.output / "deepfm.pt")
    np.savez_compressed(args.output / "predictions.npz", row_id=valid["row_id"].astype(str).to_numpy(), y=y_valid, prediction=prediction, select=mask_file["select"], evaluate=mask_file["evaluate"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
