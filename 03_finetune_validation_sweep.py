#!/usr/bin/env python3
"""
Validation-only finetuning sweep for the final model family:

    multi-seed LightGCN + recent popularity, RRF blend

This script does NOT create submissions. It trains on a validation context and
evaluates Recall@10 on the validation set.

Default validation:
    If test.csv rows are exact rows in train.csv:
        train_context = train.csv - exact test.csv rows
        valid = test.csv
    Else:
        chronological holdout by timestamp

It is designed for targeted sweeps over:
    - alpha
    - recent-popularity half-life
    - LightGCN capacity, by running the script with different dim/layers/epochs
    - seed groups

Outputs:
    outputs/lightgcn_finetune_validation_{run_id}/results.csv
    outputs/lightgcn_finetune_validation_{run_id}/results_sorted.csv
    outputs/lightgcn_finetune_validation_{run_id}/config.json
    outputs/lightgcn_finetune_validation_{run_id}/eda.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


MS_PER_DAY = 1000 * 60 * 60 * 24
KEY_COLS = ["user_id", "item_id", "timestamp"]


# -----------------------
# Utilities
# -----------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_csv(data_dir: Path, canonical: str, required: bool = True) -> Optional[Path]:
    p = data_dir / canonical
    if p.exists():
        return p
    stem = canonical.replace(".csv", "")
    matches = sorted(data_dir.glob(f"{stem}*.csv"))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find {canonical} or {stem}*.csv under {data_dir}")
    return None


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def make_interaction_matrix(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
    weights: Optional[np.ndarray] = None,
    binary_after_sum: bool = True,
) -> sparse.csr_matrix:
    rows = df["user_idx"].to_numpy(np.int32)
    cols = df["item_idx"].to_numpy(np.int32)
    vals = np.ones(len(df), dtype=np.float32) if weights is None else weights.astype(np.float32)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    X.sum_duplicates()
    if binary_after_sum:
        X.data[:] = 1.0
    return X


def dense_rrf(scores: np.ndarray, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e30, posinf=1e30, neginf=-1e30)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(scores.shape[1], dtype=np.int32)[None, :]
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)


def vector_rrf(scores: np.ndarray, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e30, posinf=1e30, neginf=-1e30)
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(len(scores), dtype=np.int32)
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)


def normalize_dense(scores: np.ndarray, method: str, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e30, posinf=1e30, neginf=-1e30)
    if method == "rrf":
        return dense_rrf(scores, rrf_k=rrf_k)
    if method == "zscore":
        mu = scores.mean(axis=1, keepdims=True)
        sd = scores.std(axis=1, keepdims=True) + 1e-6
        return ((scores - mu) / sd).astype(np.float32)
    if method == "minmax":
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
        return ((scores - lo) / (hi - lo + 1e-6)).astype(np.float32)
    if method == "none":
        return scores.astype(np.float32)
    raise ValueError(f"Unknown norm: {method}")


def normalize_vector(scores: np.ndarray, method: str, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e30, posinf=1e30, neginf=-1e30)
    if method == "rrf":
        return vector_rrf(scores, rrf_k=rrf_k)
    if method == "zscore":
        return ((scores - scores.mean()) / (scores.std() + 1e-6)).astype(np.float32)
    if method == "minmax":
        return ((scores - scores.min()) / (scores.max() - scores.min() + 1e-6)).astype(np.float32)
    if method == "none":
        return scores.astype(np.float32)
    raise ValueError(f"Unknown norm: {method}")


def time_weighted_item_scores(train_df: pd.DataFrame, n_items: int, half_life_days: int) -> np.ndarray:
    max_ts = train_df["timestamp"].max()
    age_days = (max_ts - train_df["timestamp"].to_numpy()) / MS_PER_DAY
    weights = np.exp(-np.log(2) * age_days / half_life_days).astype(np.float32)
    counts = np.bincount(
        train_df["item_idx"].to_numpy(np.int32),
        weights=weights,
        minlength=n_items,
    ).astype(np.float32)
    return np.log1p(counts).astype(np.float32)


# -----------------------
# Data
# -----------------------

@dataclass
class EncodedData:
    train_all: pd.DataFrame
    train_context: pd.DataFrame
    valid: pd.DataFrame
    sample_submission: Optional[pd.DataFrame]
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    idx2user: Dict[int, int]
    idx2item: Dict[int, int]
    n_users: int
    n_items: int
    train_item_indices: np.ndarray
    validation_name: str


def add_indices(df: pd.DataFrame, user2idx: Dict[int, int], item2idx: Dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["user_idx"] = out["user_id"].map(user2idx).astype("int32")
    out["item_idx"] = out["item_id"].map(item2idx).astype("int32")
    return out


def split_provided_test(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    valid_keys = test[KEY_COLS].drop_duplicates().assign(_valid_row=1)
    overlap = (
        test[KEY_COLS].drop_duplicates()
        .merge(train[KEY_COLS].drop_duplicates().assign(_in_train=1), on=KEY_COLS, how="left")["_in_train"]
        .fillna(0)
        .mean()
    )
    merged = train.merge(valid_keys, on=KEY_COLS, how="left")
    train_context = merged[merged["_valid_row"].isna()].drop(columns=["_valid_row"]).copy()
    valid = test.copy()
    warm_users = set(train_context.user_idx.unique())
    valid = valid[valid.user_idx.isin(warm_users)].copy()
    return train_context, valid, float(overlap)


def split_temporal_fraction(train: pd.DataFrame, valid_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = train["timestamp"].quantile(1.0 - valid_fraction)
    train_context = train[train["timestamp"] < cutoff].copy()
    valid = train[train["timestamp"] >= cutoff].copy()
    warm_users = set(train_context.user_idx.unique())
    valid = valid[valid.user_idx.isin(warm_users)].copy()
    return train_context, valid


def load_encoded_data(data_dir: Path, validation: str, valid_fraction: float) -> EncodedData:
    train_path = find_csv(data_dir, "train.csv", required=True)
    test_path = find_csv(data_dir, "test.csv", required=True)
    sample_path = find_csv(data_dir, "sample_submission.csv", required=False)
    item_meta_path = find_csv(data_dir, "item_meta.csv", required=False)

    train_raw = pd.read_csv(train_path).drop_duplicates(KEY_COLS).copy()
    test_raw = pd.read_csv(test_path).drop_duplicates(KEY_COLS).copy()
    sample = pd.read_csv(sample_path) if sample_path is not None else None
    item_meta = pd.read_csv(item_meta_path) if item_meta_path is not None else None

    train_raw["timestamp"] = pd.to_numeric(train_raw["timestamp"], errors="coerce").astype("int64")
    test_raw["timestamp"] = pd.to_numeric(test_raw["timestamp"], errors="coerce").astype("int64")

    all_user_ids = set(train_raw.user_id.unique()) | set(test_raw.user_id.unique())
    if sample is not None and "user_id" in sample.columns:
        all_user_ids |= set(sample.user_id.unique())

    all_item_ids = set(train_raw.item_id.unique()) | set(test_raw.item_id.unique())
    if item_meta is not None and "item_id" in item_meta.columns:
        all_item_ids |= set(item_meta.item_id.unique())

    all_user_ids = np.array(sorted(int(u) for u in all_user_ids))
    all_item_ids = np.array(sorted(int(i) for i in all_item_ids))

    user2idx = {int(u): i for i, u in enumerate(all_user_ids)}
    item2idx = {int(it): i for i, it in enumerate(all_item_ids)}
    idx2user = {i: int(u) for u, i in user2idx.items()}
    idx2item = {i: int(it) for it, i in item2idx.items()}

    train = add_indices(train_raw, user2idx, item2idx)
    test = add_indices(test_raw, user2idx, item2idx)

    if sample is not None and "user_id" in sample.columns:
        sample["user_idx"] = sample["user_id"].map(user2idx).astype("int32")

    if validation == "provided_test":
        train_context, valid, overlap = split_provided_test(train, test)
        if overlap < 0.95:
            print(f"WARNING: provided test overlap only {overlap:.2%}; using temporal split.")
            train_context, valid = split_temporal_fraction(train, valid_fraction=valid_fraction)
            validation_name = f"temporal_fraction_{valid_fraction}"
        else:
            validation_name = "provided_test_exact_rows_removed"
    else:
        train_context, valid = split_temporal_fraction(train, valid_fraction=valid_fraction)
        validation_name = f"temporal_fraction_{valid_fraction}"

    train_item_indices = np.array(sorted(train_context.item_idx.unique()), dtype=np.int32)

    return EncodedData(
        train_all=train,
        train_context=train_context,
        valid=valid,
        sample_submission=sample,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2user=idx2user,
        idx2item=idx2item,
        n_users=len(all_user_ids),
        n_items=len(all_item_ids),
        train_item_indices=train_item_indices,
        validation_name=validation_name,
    )


# -----------------------
# LightGCN
# -----------------------

class TorchLightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self, norm_adj: torch.Tensor):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        final = torch.stack(embs, dim=0).mean(dim=0)
        return torch.split(final, [self.n_users, self.n_items], dim=0)


def build_norm_adj(X_graph: sparse.csr_matrix, n_users: int, n_items: int, device: torch.device) -> torch.Tensor:
    X = X_graph.tocoo()
    u = X.row.astype(np.int64)
    i = X.col.astype(np.int64) + n_users
    w = X.data.astype(np.float32)

    rows = np.concatenate([u, i])
    cols = np.concatenate([i, u])
    vals_raw = np.concatenate([w, w])

    deg = np.bincount(rows, weights=vals_raw, minlength=n_users + n_items).astype(np.float32)
    deg_inv_sqrt = np.power(deg + 1e-8, -0.5)
    vals = vals_raw * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    idx = torch.LongTensor(np.vstack([rows, cols]))
    vals_t = torch.FloatTensor(vals)
    adj = torch.sparse_coo_tensor(idx, vals_t, size=(n_users + n_items, n_users + n_items))
    return adj.coalesce().to(device)


def prepare_user_positive_arrays(X_binary: sparse.csr_matrix):
    X = X_binary.tocsr()
    active_users = np.where(np.diff(X.indptr) > 0)[0].astype(np.int32)
    pos_arrays = {}
    pos_sets = {}
    for u in active_users:
        arr = X[u].indices.astype(np.int32)
        pos_arrays[int(u)] = arr
        pos_sets[int(u)] = set(arr.tolist())
    return active_users, pos_arrays, pos_sets


def sample_bpr_batch(
    active_users: np.ndarray,
    pos_arrays: Dict[int, np.ndarray],
    pos_sets: Dict[int, set],
    n_items: int,
    batch_size: int,
    rng: np.random.Generator,
):
    users = rng.choice(active_users, size=batch_size, replace=True)
    pos = np.empty(batch_size, dtype=np.int64)
    for r, u in enumerate(users):
        arr = pos_arrays[int(u)]
        pos[r] = arr[rng.integers(0, len(arr))]

    neg = rng.integers(0, n_items, size=batch_size, dtype=np.int64)
    for r, u in enumerate(users):
        seen = pos_sets[int(u)]
        tries = 0
        while int(neg[r]) in seen and tries < 100:
            neg[r] = rng.integers(0, n_items)
            tries += 1

    return users.astype(np.int64), pos, neg


@dataclass
class TrainConfig:
    dim: int
    layers: int
    epochs: int
    batch_size: int
    lr: float
    reg: float


@dataclass
class Embeddings:
    seed: int
    user_emb: np.ndarray
    item_emb: np.ndarray


def train_lightgcn(
    seed: int,
    cfg: TrainConfig,
    X_binary: sparse.csr_matrix,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> Embeddings:
    seed_everything(seed)
    print(f"\nTraining LightGCN seed={seed} dim={cfg.dim} layers={cfg.layers} epochs={cfg.epochs}")

    active_users, pos_arrays, pos_sets = prepare_user_positive_arrays(X_binary)
    norm_adj = build_norm_adj(X_binary, n_users, n_items, device)

    model = TorchLightGCN(n_users, n_items, cfg.dim, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, X_binary.nnz // cfg.batch_size)

    for epoch in tqdm(range(1, cfg.epochs + 1), desc=f"LightGCN seed={seed}"):
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            users, pos, neg = sample_bpr_batch(active_users, pos_arrays, pos_sets, n_items, cfg.batch_size, rng)

            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos).to(device)
            neg_t = torch.LongTensor(neg).to(device)

            user_e, item_e = model.propagate(norm_adj)
            u_e = user_e[users_t]
            p_e = item_e[pos_t]
            n_e = item_e[neg_t]

            pos_scores = (u_e * p_e).sum(dim=1)
            neg_scores = (u_e * n_e).sum(dim=1)
            mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss = cfg.reg * (
                model.user_emb(users_t).norm(2).pow(2)
                + model.item_emb(pos_t).norm(2).pow(2)
                + model.item_emb(neg_t).norm(2).pow(2)
            ) / cfg.batch_size
            loss = mf_loss + reg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())

        if epoch == 1 or epoch % 20 == 0 or epoch == cfg.epochs:
            print(f"seed={seed} epoch {epoch:03d}/{cfg.epochs}, loss={total_loss / steps_per_epoch:.5f}")

    model.eval()
    with torch.no_grad():
        user_e, item_e = model.propagate(norm_adj)
        user_emb = user_e.detach().cpu().numpy().astype(np.float32)
        item_emb = item_e.detach().cpu().numpy().astype(np.float32)

    del model, norm_adj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return Embeddings(seed=seed, user_emb=user_emb, item_emb=item_emb)


# -----------------------
# Evaluation
# -----------------------

def build_truth(valid_df: pd.DataFrame) -> Dict[int, set]:
    return valid_df.groupby("user_idx")["item_idx"].apply(lambda x: set(int(v) for v in x.values)).to_dict()


def build_lightgcn_component(
    embeddings: Sequence[Embeddings],
    user_indices: np.ndarray,
    norm: str,
    rrf_k: float,
) -> np.ndarray:
    out = None
    for emb in embeddings:
        raw = emb.user_emb[user_indices] @ emb.item_emb.T
        comp = normalize_dense(raw, method=norm, rrf_k=rrf_k)
        out = comp if out is None else out + comp
    out /= float(len(embeddings))
    return out.astype(np.float32)


def recall_at_k_blend(
    embeddings: Sequence[Embeddings],
    pop_component: np.ndarray,
    alpha: float,
    valid_df: pd.DataFrame,
    X_seen: sparse.csr_matrix,
    allowed_items: np.ndarray,
    users: Sequence[int],
    norm: str,
    rrf_k: float,
    k: int,
    batch_size: int,
    desc: Optional[str] = None,
) -> float:
    truth = build_truth(valid_df)
    users = np.array([int(u) for u in users if int(u) in truth], dtype=np.int32)
    if len(users) == 0:
        return float("nan")

    allowed_items = np.asarray(allowed_items, dtype=np.int32)
    recalls = []

    iterator = range(0, len(users), batch_size)
    if desc:
        iterator = tqdm(iterator, desc=desc, leave=False)

    for start in iterator:
        batch_users = users[start:start + batch_size]
        lgcn_component = build_lightgcn_component(embeddings, batch_users, norm=norm, rrf_k=rrf_k)
        scores = (1.0 - alpha) * lgcn_component + alpha * pop_component[None, :]

        scores_allowed = scores[:, allowed_items].copy()

        for r, u in enumerate(batch_users):
            seen = X_seen[int(u)].indices
            if len(seen):
                seen_mask = np.isin(allowed_items, seen, assume_unique=False)
                scores_allowed[r, seen_mask] = -np.inf

            row = scores_allowed[r]
            finite = np.isfinite(row)
            if not np.any(finite):
                recalls.append(0.0)
                continue

            kk = min(k, finite.sum())
            candidate_pos = np.where(finite)[0]
            candidate_scores = row[candidate_pos]

            if len(candidate_pos) > kk:
                part = np.argpartition(-candidate_scores, kk - 1)[:kk]
                top_local = candidate_pos[part]
                top_scores = candidate_scores[part]
                top_items = allowed_items[top_local[np.argsort(-top_scores)]]
            else:
                top_items = allowed_items[candidate_pos[np.argsort(-candidate_scores)]]

            t = truth.get(int(u), set())
            recalls.append(len(set(int(x) for x in top_items[:k]) & t) / min(k, len(t)))

    return float(np.mean(recalls)) if recalls else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--validation", type=str, default="provided_test", choices=["provided_test", "temporal"])
    parser.add_argument("--valid_fraction", type=float, default=0.15)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.0002)
    parser.add_argument("--seeds", type=str, default="42,43,44")

    parser.add_argument("--recent_half_lives", type=str, default="120,150,180,210,240")
    parser.add_argument("--alphas", type=str, default="0.45,0.475,0.5,0.525,0.55")
    parser.add_argument("--blend_norm", type=str, default="rrf", choices=["rrf", "zscore", "minmax", "none"])
    parser.add_argument("--rrf_k", type=float, default=60.0)

    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_eval_users", type=int, default=None)
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    if args.quick:
        print("Running in --quick mode.")
        args.dim = min(args.dim, 128)
        args.epochs = min(args.epochs, 30)
        args.seeds = args.seeds.split(",")[0]
        args.max_eval_users = args.max_eval_users or 1000

    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_id = args.run_id or now_run_id()
    out_dir = Path(args.output_root) / f"lightgcn_finetune_validation_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Output dir:", out_dir.resolve())
    print("Device:", device)

    data = load_encoded_data(Path(args.data_dir), validation=args.validation, valid_fraction=args.valid_fraction)
    X_context = make_interaction_matrix(data.train_context, data.n_users, data.n_items, binary_after_sum=True)

    all_valid_users = np.array(sorted(data.valid.user_idx.unique()), dtype=np.int32)
    if args.max_eval_users is not None and len(all_valid_users) > args.max_eval_users:
        rng = np.random.default_rng(args.seed)
        all_valid_users = np.array(sorted(rng.choice(all_valid_users, size=args.max_eval_users, replace=False)), dtype=np.int32)

    target_valid_users = None
    if data.sample_submission is not None and "user_idx" in data.sample_submission.columns:
        target_valid_users = np.array(sorted(set(all_valid_users) & set(data.sample_submission.user_idx.astype(int))), dtype=np.int32)

    eda = {
        "validation_name": data.validation_name,
        "n_users": data.n_users,
        "n_items": data.n_items,
        "train_all_rows": int(len(data.train_all)),
        "train_context_rows": int(len(data.train_context)),
        "valid_rows": int(len(data.valid)),
        "valid_users": int(data.valid.user_idx.nunique()),
        "eval_users": int(len(all_valid_users)),
        "target_overlap_eval_users": int(len(target_valid_users)) if target_valid_users is not None else 0,
        "train_context_items": int(len(data.train_item_indices)),
    }
    with open(out_dir / "eda.json", "w") as f:
        json.dump(eda, f, indent=2)
    print(json.dumps(eda, indent=2))

    seeds = parse_int_list(args.seeds)
    half_lives = parse_int_list(args.recent_half_lives)
    alphas = parse_float_list(args.alphas)

    cfg = TrainConfig(
        dim=args.dim,
        layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
    )

    embeddings = []
    for seed in seeds:
        embeddings.append(
            train_lightgcn(
                seed=seed,
                cfg=cfg,
                X_binary=X_context,
                n_users=data.n_users,
                n_items=data.n_items,
                device=device,
            )
        )

    rows = []
    pop_components = {
        hl: normalize_vector(
            time_weighted_item_scores(data.train_context, data.n_items, hl),
            method=args.blend_norm,
            rrf_k=args.rrf_k,
        )
        for hl in half_lives
    }

    for hl in half_lives:
        for alpha in alphas:
            t0 = time.time()
            name = f"lgcn_d{args.dim}_l{args.layers}_e{args.epochs}_seeds{'-'.join(map(str, seeds))}_hl{hl}_alpha{alpha:g}"
            recall_all = recall_at_k_blend(
                embeddings=embeddings,
                pop_component=pop_components[hl],
                alpha=alpha,
                valid_df=data.valid,
                X_seen=X_context,
                allowed_items=data.train_item_indices,
                users=all_valid_users,
                norm=args.blend_norm,
                rrf_k=args.rrf_k,
                k=args.k,
                batch_size=args.eval_batch_size,
                desc=name,
            )
            recall_target = np.nan
            if target_valid_users is not None and len(target_valid_users):
                recall_target = recall_at_k_blend(
                    embeddings=embeddings,
                    pop_component=pop_components[hl],
                    alpha=alpha,
                    valid_df=data.valid,
                    X_seen=X_context,
                    allowed_items=data.train_item_indices,
                    users=target_valid_users,
                    norm=args.blend_norm,
                    rrf_k=args.rrf_k,
                    k=args.k,
                    batch_size=args.eval_batch_size,
                    desc=name + ":target",
                )

            row = {
                "experiment": name,
                "dim": args.dim,
                "layers": args.layers,
                "epochs": args.epochs,
                "seeds": ",".join(map(str, seeds)),
                "recent_half_life": hl,
                "alpha": alpha,
                "recall_at_10_all_valid_users": recall_all,
                "recall_at_10_target_overlap_users": recall_target,
                "time_sec": time.time() - t0,
            }
            rows.append(row)
            print(f"{name:80s} | all={recall_all:.6f} | target={recall_target:.6f} | {row['time_sec']:.1f}s")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)

    sorted_all = results.sort_values(
        ["recall_at_10_all_valid_users", "recall_at_10_target_overlap_users"],
        ascending=[False, False],
        na_position="last",
    )
    sorted_target = results.sort_values(
        ["recall_at_10_target_overlap_users", "recall_at_10_all_valid_users"],
        ascending=[False, False],
        na_position="last",
    )

    sorted_all.to_csv(out_dir / "results_sorted.csv", index=False)
    sorted_target.to_csv(out_dir / "results_sorted_target.csv", index=False)

    print("\n=== Top by all-valid Recall@10 ===")
    print(sorted_all.head(25).to_string(index=False))

    print("\n=== Top by target-overlap Recall@10 ===")
    print(sorted_target.head(25).to_string(index=False))

    print("\nSaved:")
    print(" ", out_dir / "results.csv")
    print(" ", out_dir / "results_sorted.csv")
    print(" ", out_dir / "results_sorted_target.csv")
    print(" ", out_dir / "config.json")
    print(" ", out_dir / "eda.json")


if __name__ == "__main__":
    main()
