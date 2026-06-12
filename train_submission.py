#!/usr/bin/env python3
"""
Train final LightGCN + recent-popularity (+ optional trend) RRF blends and write Kaggle submissions.

This script is for final submission generation, not validation. It trains on all
deduplicated train.csv interactions and writes one submission CSV per requested
alpha / recency half-life combination.

Recommended model family from experiments:
    LightGCN + recent popularity, combined by RRF rank blending.

Expected files under --data_dir:
    train.csv
    sample_submission.csv

Optional files:
    test.csv          ignored by default
    item_meta.csv     ignored here, because metadata was not helpful enough
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_trend_specs(s: str) -> List[Tuple[int, int]]:
    """
    Parse trend specs like "120:365,90:365" into [(120, 365), (90, 365)].
    """
    if s is None or str(s).strip() == "":
        return []
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        short_hl, long_hl = part.split(":")
        out.append((int(short_hl), int(long_hl)))
    return out


def safe_name(x) -> str:
    return str(x).replace(".", "p").replace("-", "m").replace(",", "_")


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


def time_decay_weights(df: pd.DataFrame, half_life_days: int) -> np.ndarray:
    max_ts = df["timestamp"].max()
    age_days = (max_ts - df["timestamp"].to_numpy()) / MS_PER_DAY
    return np.exp(-np.log(2) * age_days / half_life_days).astype(np.float32)


def time_weighted_counts(
    train_df: pd.DataFrame,
    n_items: int,
    half_life_days: int,
) -> np.ndarray:
    max_ts = train_df["timestamp"].max()
    age_days = (max_ts - train_df["timestamp"].to_numpy()) / MS_PER_DAY
    weights = np.exp(-np.log(2) * age_days / half_life_days).astype(np.float32)
    return np.bincount(
        train_df["item_idx"].to_numpy(np.int32),
        weights=weights,
        minlength=n_items,
    ).astype(np.float32)


def time_weighted_item_scores(
    train_df: pd.DataFrame,
    n_items: int,
    half_life_days: int,
) -> np.ndarray:
    return np.log1p(
        time_weighted_counts(train_df, n_items=n_items, half_life_days=half_life_days)
    ).astype(np.float32)


def trend_score(
    train_df: pd.DataFrame,
    n_items: int,
    short_hl: int,
    long_hl: int,
) -> np.ndarray:
    """
    Trend acceleration score used in validation:
        log(1 + short_count) - log(1 + long_count) + 0.15 * log(1 + short_count)

    The support term prevents extremely rare one-off items from dominating.
    """
    short = time_weighted_counts(train_df, n_items=n_items, half_life_days=short_hl)
    long = time_weighted_counts(train_df, n_items=n_items, half_life_days=long_hl)
    return (
        np.log1p(short) - np.log1p(long + 1e-6) + 0.15 * np.log1p(short)
    ).astype(np.float32)


def dense_rrf(scores: np.ndarray, rrf_k: float = 60.0) -> np.ndarray:
    """
    Convert dense scores to reciprocal-rank features per row.
    Higher input score -> lower rank -> higher RRF value.
    """
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=-1e30, posinf=1e30, neginf=-1e30)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(scores.shape[1], dtype=np.int32)[None, :]
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)


def vector_rrf(scores: np.ndarray, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=-1e30, posinf=1e30, neginf=-1e30)
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(len(scores), dtype=np.int32)
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)


def normalize_dense(scores: np.ndarray, method: str, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e30, posinf=1e30, neginf=-1e30)
    if method == "none":
        return scores
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
    raise ValueError(f"Unknown blend_norm: {method}")


def normalize_vector(scores: np.ndarray, method: str, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e30, posinf=1e30, neginf=-1e30)
    if method == "none":
        return scores.astype(np.float32)
    if method == "rrf":
        return vector_rrf(scores, rrf_k=rrf_k)
    if method == "zscore":
        return ((scores - scores.mean()) / (scores.std() + 1e-6)).astype(np.float32)
    if method == "minmax":
        return ((scores - scores.min()) / (scores.max() - scores.min() + 1e-6)).astype(np.float32)
    raise ValueError(f"Unknown blend_norm: {method}")


def parse_dynamic_alpha_spec(spec: str) -> List[Tuple[int, float]]:
    """
    Format: upper_bound:alpha,upper_bound:alpha,...
    Example: 3:0.75,6:0.65,12:0.55,999999:0.45
    Means:
      history_len <= 3 -> 0.75
      history_len <= 6 -> 0.65
      etc.
    """
    bins = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        upper, alpha = part.split(":")
        bins.append((int(upper), float(alpha)))
    bins.sort(key=lambda x: x[0])
    if not bins:
        raise ValueError("Empty dynamic alpha spec.")
    return bins


def alpha_for_history_len(hist_len: int, bins: Sequence[Tuple[int, float]]) -> float:
    for upper, alpha in bins:
        if hist_len <= upper:
            return alpha
    return bins[-1][1]


# -----------------------
# Data
# -----------------------

@dataclass
class EncodedData:
    train: pd.DataFrame
    sample_submission: pd.DataFrame
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    idx2user: Dict[int, int]
    idx2item: Dict[int, int]
    n_users: int
    n_items: int
    train_item_indices: np.ndarray


def load_data(data_dir: Path) -> EncodedData:
    train_path = find_csv(data_dir, "train.csv", required=True)
    sample_path = find_csv(data_dir, "sample_submission.csv", required=True)

    train_raw = pd.read_csv(train_path).drop_duplicates(KEY_COLS).copy()
    sample = pd.read_csv(sample_path).copy()

    train_raw["timestamp"] = pd.to_numeric(train_raw["timestamp"], errors="coerce").astype("int64")

    all_user_ids = set(train_raw.user_id.unique()) | set(sample.user_id.unique())
    all_item_ids = set(train_raw.item_id.unique())

    # Include item IDs from the sample item_id column only if they are parseable and present,
    # but recommendations are restricted to train items.
    all_user_ids = np.array(sorted(int(u) for u in all_user_ids))
    all_item_ids = np.array(sorted(int(i) for i in all_item_ids))

    user2idx = {int(u): i for i, u in enumerate(all_user_ids)}
    item2idx = {int(it): i for i, it in enumerate(all_item_ids)}
    idx2user = {i: int(u) for u, i in user2idx.items()}
    idx2item = {i: int(it) for it, i in item2idx.items()}

    train = train_raw.copy()
    train["user_idx"] = train["user_id"].map(user2idx).astype("int32")
    train["item_idx"] = train["item_id"].map(item2idx).astype("int32")

    sample["user_idx"] = sample["user_id"].map(user2idx)
    if sample["user_idx"].isna().any():
        missing = int(sample["user_idx"].isna().sum())
        raise ValueError(f"{missing} sample_submission users are not in the encoded user universe.")
    sample["user_idx"] = sample["user_idx"].astype("int32")

    train_item_indices = np.array(sorted(train.item_idx.unique()), dtype=np.int32)

    return EncodedData(
        train=train,
        sample_submission=sample,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2user=idx2user,
        idx2item=idx2item,
        n_users=len(all_user_ids),
        n_items=len(all_item_ids),
        train_item_indices=train_item_indices,
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


def build_popularity_sampler(X_binary: sparse.csr_matrix, power: float = 0.75) -> np.ndarray:
    counts = np.asarray(X_binary.sum(axis=0)).ravel().astype(np.float64)
    probs = np.power(counts + 1e-6, power)
    probs /= probs.sum()
    return probs


def sample_bpr_batch(
    active_users: np.ndarray,
    pos_arrays: Dict[int, np.ndarray],
    pos_sets: Dict[int, set],
    n_items: int,
    batch_size: int,
    rng: np.random.Generator,
    neg_probs: Optional[np.ndarray] = None,
):
    users = rng.choice(active_users, size=batch_size, replace=True)
    pos = np.empty(batch_size, dtype=np.int64)

    for r, u in enumerate(users):
        positives = pos_arrays[int(u)]
        pos[r] = positives[rng.integers(0, len(positives))]

    if neg_probs is None:
        neg = rng.integers(0, n_items, size=batch_size, dtype=np.int64)
    else:
        neg = rng.choice(np.arange(n_items), size=batch_size, replace=True, p=neg_probs).astype(np.int64)

    for r, u in enumerate(users):
        seen = pos_sets[int(u)]
        tries = 0
        while int(neg[r]) in seen and tries < 100:
            if neg_probs is None:
                neg[r] = rng.integers(0, n_items)
            else:
                neg[r] = rng.choice(np.arange(n_items), p=neg_probs)
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
    neg_sampling: str
    time_decay_graph_half_life: Optional[int]


@dataclass
class TrainedEmbeddings:
    seed: int
    user_emb: np.ndarray
    item_emb: np.ndarray


def train_one_lightgcn(
    seed: int,
    cfg: TrainConfig,
    X_binary: sparse.csr_matrix,
    X_graph: sparse.csr_matrix,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> TrainedEmbeddings:
    seed_everything(seed)
    print(f"\nTraining LightGCN seed={seed} dim={cfg.dim} layers={cfg.layers} epochs={cfg.epochs} neg={cfg.neg_sampling}")

    active_users, pos_arrays, pos_sets = prepare_user_positive_arrays(X_binary)
    neg_probs = build_popularity_sampler(X_binary) if cfg.neg_sampling == "popularity" else None
    norm_adj = build_norm_adj(X_graph, n_users, n_items, device)

    model = TorchLightGCN(n_users, n_items, cfg.dim, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, X_binary.nnz // cfg.batch_size)

    for epoch in tqdm(range(1, cfg.epochs + 1), desc=f"LightGCN seed={seed}"):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            users, pos, neg = sample_bpr_batch(
                active_users, pos_arrays, pos_sets, n_items, cfg.batch_size, rng, neg_probs=neg_probs
            )

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

    return TrainedEmbeddings(seed=seed, user_emb=user_emb, item_emb=item_emb)


# -----------------------
# Submission generation
# -----------------------

def make_lightgcn_component(
    embeddings: Sequence[TrainedEmbeddings],
    user_indices: np.ndarray,
    blend_norm: str,
    rrf_k: float,
) -> np.ndarray:
    """
    Returns a normalized LightGCN component for this user batch.
    For multiple seeds, average normalized components seed-wise.
    """
    out = None
    for emb in embeddings:
        raw = emb.user_emb[user_indices] @ emb.item_emb.T
        comp = normalize_dense(raw, method=blend_norm, rrf_k=rrf_k)
        out = comp if out is None else out + comp
    out /= float(len(embeddings))
    return out.astype(np.float32)


def build_submission_dataframe(
    sample: pd.DataFrame,
    pred_item_indices: Dict[int, List[int]],
    idx2item: Dict[int, int],
) -> pd.DataFrame:
    rows = []

    has_id = "ID" in sample.columns
    if "prediction" in sample.columns:
        pred_col = "prediction"
        sep = " "
    else:
        pred_col = "item_id"
        sep = ","

    for _, row in sample.iterrows():
        uidx = int(row["user_idx"])
        item_ids = [str(idx2item[int(i)]) for i in pred_item_indices[uidx]]

        out = {}
        if has_id:
            out["ID"] = row["ID"]
        out["user_id"] = int(row["user_id"])
        out[pred_col] = sep.join(item_ids)
        rows.append(out)

    return pd.DataFrame(rows)


def generate_predictions(
    data: EncodedData,
    X_seen: sparse.csr_matrix,
    embeddings: Sequence[TrainedEmbeddings],
    pop_score_vectors: Dict[int, np.ndarray],
    alphas: Sequence[float],
    blend_norm: str,
    rrf_k: float,
    batch_size: int,
    output_dir: Path,
    name_prefix: str,
    dynamic_alpha_bins: Optional[List[Tuple[int, float]]] = None,
    trend_score_vectors: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    trend_gammas: Optional[Sequence[float]] = None,
    trend_base_alphas: Optional[Sequence[float]] = None,
) -> List[Path]:
    sample = data.sample_submission
    target_users = sample["user_idx"].to_numpy(np.int32)
    allowed_items = data.train_item_indices
    allowed_mask = np.zeros(data.n_items, dtype=bool)
    allowed_mask[allowed_items] = True

    # User history lengths for dynamic alpha.
    hist_len = np.diff(X_seen.indptr)

    output_paths = []

    # Precompute normalized popularity components.
    pop_components = {
        hl: normalize_vector(scores, method=blend_norm, rrf_k=rrf_k)
        for hl, scores in pop_score_vectors.items()
    }

    trend_components = {}
    if trend_score_vectors:
        trend_components = {
            spec: normalize_vector(scores, method=blend_norm, rrf_k=rrf_k)
            for spec, scores in trend_score_vectors.items()
        }

    # Fixed alpha submissions.
    for half_life, pop_component in pop_components.items():
        for alpha in alphas:
            print(f"\nGenerating submission: half_life={half_life}, alpha={alpha}")
            pred_item_indices: Dict[int, List[int]] = {}

            for start in tqdm(range(0, len(target_users), batch_size), desc=f"predict hl={half_life} a={alpha}", leave=False):
                batch_users = target_users[start:start + batch_size]
                lgcn_component = make_lightgcn_component(embeddings, batch_users, blend_norm=blend_norm, rrf_k=rrf_k)
                scores = (1.0 - alpha) * lgcn_component + alpha * pop_component[None, :]

                # Restrict to train-interaction items and exclude seen items.
                scores[:, ~allowed_mask] = -np.inf
                for r, u in enumerate(batch_users):
                    seen = X_seen[int(u)].indices
                    if len(seen):
                        scores[r, seen] = -np.inf

                    row = scores[r]
                    k = 10
                    finite = np.isfinite(row)
                    if finite.sum() < k:
                        raise RuntimeError(f"User {u} has fewer than {k} finite candidate scores.")
                    candidate_idx = np.where(finite)[0]
                    candidate_scores = row[candidate_idx]
                    part = np.argpartition(-candidate_scores, k - 1)[:k]
                    top_items = candidate_idx[part[np.argsort(-candidate_scores[part])]]
                    pred_item_indices[int(u)] = [int(x) for x in top_items]

            sub = build_submission_dataframe(sample, pred_item_indices, data.idx2item)
            out_path = output_dir / f"{name_prefix}_hl{half_life}_alpha{safe_name(alpha)}.csv"
            sub.to_csv(out_path, index=False)
            print("Saved", out_path)
            output_paths.append(out_path)

    # Dynamic alpha submissions.
    if dynamic_alpha_bins is not None:
        for half_life, pop_component in pop_components.items():
            print(f"\nGenerating dynamic-alpha submission: half_life={half_life}, bins={dynamic_alpha_bins}")
            pred_item_indices: Dict[int, List[int]] = {}

            for start in tqdm(range(0, len(target_users), batch_size), desc=f"predict dynamic hl={half_life}", leave=False):
                batch_users = target_users[start:start + batch_size]
                lgcn_component = make_lightgcn_component(embeddings, batch_users, blend_norm=blend_norm, rrf_k=rrf_k)
                user_alphas = np.array(
                    [alpha_for_history_len(int(hist_len[int(u)]), dynamic_alpha_bins) for u in batch_users],
                    dtype=np.float32,
                )
                scores = (1.0 - user_alphas[:, None]) * lgcn_component + user_alphas[:, None] * pop_component[None, :]

                scores[:, ~allowed_mask] = -np.inf
                for r, u in enumerate(batch_users):
                    seen = X_seen[int(u)].indices
                    if len(seen):
                        scores[r, seen] = -np.inf

                    row = scores[r]
                    k = 10
                    finite = np.isfinite(row)
                    if finite.sum() < k:
                        raise RuntimeError(f"User {u} has fewer than {k} finite candidate scores.")
                    candidate_idx = np.where(finite)[0]
                    candidate_scores = row[candidate_idx]
                    part = np.argpartition(-candidate_scores, k - 1)[:k]
                    top_items = candidate_idx[part[np.argsort(-candidate_scores[part])]]
                    pred_item_indices[int(u)] = [int(x) for x in top_items]

            sub = build_submission_dataframe(sample, pred_item_indices, data.idx2item)
            spec_name = "_".join([f"le{u}_a{safe_name(a)}" for u, a in dynamic_alpha_bins])
            out_path = output_dir / f"{name_prefix}_hl{half_life}_dynamic_{spec_name}.csv"
            sub.to_csv(out_path, index=False)
            print("Saved", out_path)
            output_paths.append(out_path)


    # Trend-augmented submissions.
    if trend_components and trend_gammas:
        base_alphas = list(trend_base_alphas) if trend_base_alphas else list(alphas)
        for half_life, pop_component in pop_components.items():
            for alpha in base_alphas:
                for (short_hl, long_hl), trend_component in trend_components.items():
                    for gamma in trend_gammas:
                        print(f"\nGenerating trend submission: hl={half_life}, alpha={alpha}, trend={short_hl}v{long_hl}, gamma={gamma}")
                        pred_item_indices: Dict[int, List[int]] = {}

                        for start in tqdm(
                            range(0, len(target_users), batch_size),
                            desc=f"predict trend {short_hl}v{long_hl} g={gamma}",
                            leave=False,
                        ):
                            batch_users = target_users[start:start + batch_size]
                            lgcn_component = make_lightgcn_component(
                                embeddings, batch_users, blend_norm=blend_norm, rrf_k=rrf_k
                            )
                            scores = (
                                (1.0 - alpha) * lgcn_component
                                + alpha * pop_component[None, :]
                                + float(gamma) * trend_component[None, :]
                            )

                            scores[:, ~allowed_mask] = -np.inf
                            for r, u in enumerate(batch_users):
                                seen = X_seen[int(u)].indices
                                if len(seen):
                                    scores[r, seen] = -np.inf

                                row = scores[r]
                                k = 10
                                finite = np.isfinite(row)
                                if finite.sum() < k:
                                    raise RuntimeError(f"User {u} has fewer than {k} finite candidate scores.")
                                candidate_idx = np.where(finite)[0]
                                candidate_scores = row[candidate_idx]
                                part = np.argpartition(-candidate_scores, k - 1)[:k]
                                top_items = candidate_idx[part[np.argsort(-candidate_scores[part])]]
                                pred_item_indices[int(u)] = [int(x) for x in top_items]

                        sub = build_submission_dataframe(sample, pred_item_indices, data.idx2item)
                        out_path = output_dir / (
                            f"{name_prefix}_hl{half_life}_alpha{safe_name(alpha)}"
                            f"_trend{short_hl}v{long_hl}_gamma{safe_name(gamma)}.csv"
                        )
                        sub.to_csv(out_path, index=False)
                        print("Saved", out_path)
                        output_paths.append(out_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated LightGCN seeds, e.g. 42,43,44.")
    parser.add_argument("--neg_sampling", type=str, default="uniform", choices=["uniform", "popularity"])
    parser.add_argument(
        "--time_decay_graph_half_life",
        type=int,
        default=0,
        help="0 means binary graph. Positive value uses recency-weighted graph propagation.",
    )

    parser.add_argument("--recent_half_lives", type=str, default="180")
    parser.add_argument("--alphas", type=str, default="0.5,0.6")
    parser.add_argument("--trend_specs", type=str, default="", help="Optional trend specs, e.g. 120:365,90:365.")
    parser.add_argument("--trend_gammas", type=str, default="", help="Optional trend weights, e.g. 0.1,0.2,0.3.")
    parser.add_argument("--trend_base_alphas", type=str, default="", help="Optional alpha list for trend submissions; defaults to --alphas.")
    parser.add_argument("--blend_norm", type=str, default="rrf", choices=["rrf", "zscore", "minmax", "none"])
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--prediction_batch_size", type=int, default=128)

    parser.add_argument("--dynamic_alpha", action="store_true")
    parser.add_argument(
        "--dynamic_alpha_spec",
        type=str,
        default="3:0.75,6:0.65,12:0.55,999999:0.45",
        help="History-length alpha bins for --dynamic_alpha.",
    )

    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    seed_everything(42)

    run_id = args.run_id or now_run_id()
    out_dir = Path(args.output_root) / f"submission_lightgcn_recent_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    data = load_data(Path(args.data_dir))
    print("Output dir:", out_dir.resolve())
    print("Device:", device)
    print(f"Users={data.n_users:,}, items={data.n_items:,}, train rows={len(data.train):,}, submission users={len(data.sample_submission):,}")

    X_binary = make_interaction_matrix(data.train, data.n_users, data.n_items, binary_after_sum=True)

    if args.time_decay_graph_half_life and args.time_decay_graph_half_life > 0:
        graph_weights = time_decay_weights(data.train, args.time_decay_graph_half_life)
        X_graph = make_interaction_matrix(
            data.train,
            data.n_users,
            data.n_items,
            weights=graph_weights,
            binary_after_sum=False,
        )
        graph_name = f"timegraph{args.time_decay_graph_half_life}"
    else:
        X_graph = X_binary
        graph_name = "binarygraph"

    seeds = parse_int_list(args.seeds)
    alphas = parse_float_list(args.alphas)
    recent_half_lives = parse_int_list(args.recent_half_lives)
    trend_specs = parse_trend_specs(args.trend_specs)
    trend_gammas = parse_float_list(args.trend_gammas)
    trend_base_alphas = parse_float_list(args.trend_base_alphas)

    cfg = TrainConfig(
        dim=args.dim,
        layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        neg_sampling=args.neg_sampling,
        time_decay_graph_half_life=args.time_decay_graph_half_life if args.time_decay_graph_half_life > 0 else None,
    )

    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "train_config": asdict(cfg), "seeds_parsed": seeds}, f, indent=2)

    embeddings = []
    t0 = time.time()
    for seed in seeds:
        emb = train_one_lightgcn(
            seed=seed,
            cfg=cfg,
            X_binary=X_binary,
            X_graph=X_graph,
            n_users=data.n_users,
            n_items=data.n_items,
            device=device,
        )
        embeddings.append(emb)

    pop_score_vectors = {
        hl: time_weighted_item_scores(data.train, n_items=data.n_items, half_life_days=hl)
        for hl in recent_half_lives
    }

    trend_score_vectors = {
        spec: trend_score(data.train, n_items=data.n_items, short_hl=spec[0], long_hl=spec[1])
        for spec in trend_specs
    }

    name_prefix = (
        f"lgcn_d{args.dim}_l{args.layers}_e{args.epochs}_{graph_name}_"
        f"{args.neg_sampling}_seeds{'-'.join(map(str, seeds))}_{args.blend_norm}"
    )

    dynamic_bins = parse_dynamic_alpha_spec(args.dynamic_alpha_spec) if args.dynamic_alpha else None

    output_paths = generate_predictions(
        data=data,
        X_seen=X_binary,
        embeddings=embeddings,
        pop_score_vectors=pop_score_vectors,
        alphas=alphas,
        blend_norm=args.blend_norm,
        rrf_k=args.rrf_k,
        batch_size=args.prediction_batch_size,
        output_dir=out_dir,
        name_prefix=name_prefix,
        dynamic_alpha_bins=dynamic_bins,
        trend_score_vectors=trend_score_vectors,
        trend_gammas=trend_gammas,
        trend_base_alphas=trend_base_alphas,
    )

    summary = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "num_outputs": len(output_paths),
        "outputs": [str(p) for p in output_paths],
        "train_time_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Outputs:")
    for p in output_paths:
        print(" ", p)
    print("Summary:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
