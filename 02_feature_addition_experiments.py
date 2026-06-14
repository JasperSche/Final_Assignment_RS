#!/usr/bin/env python3
from __future__ import annotations
import argparse
import gc
import json
import math
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
KEY_COLS = ['user_id', 'item_id', 'timestamp']

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def find_csv(data_dir: Path, canonical: str, required: bool=True) -> Optional[Path]:
    p = data_dir / canonical
    if p.exists():
        return p
    stem = canonical.replace('.csv', '')
    matches = sorted(data_dir.glob(f'{stem}*.csv'))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f'Could not find {canonical} or {stem}*.csv under {data_dir}')
    return None

def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == '':
        return []
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]

def parse_float_list(s: str) -> List[float]:
    if s is None or str(s).strip() == '':
        return []
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]

def parse_trend_specs(s: str) -> List[Tuple[int, int]]:
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(':')
        out.append((int(a), int(b)))
    return out

def parse_dynamic_alpha_specs(s: str) -> List[List[Tuple[int, float]]]:
    specs = []
    for spec in str(s).split(';'):
        spec = spec.strip()
        if not spec:
            continue
        bins = []
        for part in spec.split(','):
            upper, alpha = part.split(':')
            bins.append((int(upper), float(alpha)))
        bins.sort(key=lambda x: x[0])
        specs.append(bins)
    return specs

def alpha_for_history_len(hist_len: int, bins: Sequence[Tuple[int, float]]) -> float:
    for upper, alpha in bins:
        if hist_len <= upper:
            return alpha
    return bins[-1][1]

def spec_to_name(spec: Sequence[Tuple[int, float]]) -> str:
    return '_'.join([f"le{u}_a{str(a).replace('.', 'p')}" for u, a in spec])

def make_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int, weights: Optional[np.ndarray]=None, binary_after_sum: bool=True) -> sparse.csr_matrix:
    rows = df['user_idx'].to_numpy(np.int32)
    cols = df['item_idx'].to_numpy(np.int32)
    vals = np.ones(len(df), dtype=np.float32) if weights is None else weights.astype(np.float32)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    X.sum_duplicates()
    if binary_after_sum:
        X.data[:] = 1.0
    return X

def dense_rrf(scores: np.ndarray, rrf_k: float=60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e+30, posinf=1e+30, neginf=-1e+30)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(scores.shape[1], dtype=np.int32)[None, :]
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)

def vector_rrf(scores: np.ndarray, rrf_k: float=60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e+30, posinf=1e+30, neginf=-1e+30)
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(len(scores), dtype=np.int32)
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)

def normalize_dense(scores: np.ndarray, method: str='rrf', rrf_k: float=60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e+30, posinf=1e+30, neginf=-1e+30)
    if method == 'none':
        return scores.astype(np.float32)
    if method == 'rrf':
        return dense_rrf(scores, rrf_k=rrf_k)
    if method == 'zscore':
        mu = scores.mean(axis=1, keepdims=True)
        sd = scores.std(axis=1, keepdims=True) + 1e-06
        return ((scores - mu) / sd).astype(np.float32)
    if method == 'minmax':
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
        return ((scores - lo) / (hi - lo + 1e-06)).astype(np.float32)
    raise ValueError(f'Unknown normalization method: {method}')

def normalize_vector(scores: np.ndarray, method: str='rrf', rrf_k: float=60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e+30, posinf=1e+30, neginf=-1e+30)
    if method == 'none':
        return scores.astype(np.float32)
    if method == 'rrf':
        return vector_rrf(scores, rrf_k=rrf_k)
    if method == 'zscore':
        return ((scores - scores.mean()) / (scores.std() + 1e-06)).astype(np.float32)
    if method == 'minmax':
        return ((scores - scores.min()) / (scores.max() - scores.min() + 1e-06)).astype(np.float32)
    raise ValueError(f'Unknown normalization method: {method}')

def time_weighted_counts(train_df: pd.DataFrame, n_items: int, half_life_days: int) -> np.ndarray:
    max_ts = train_df['timestamp'].max()
    age_days = (max_ts - train_df['timestamp'].to_numpy()) / MS_PER_DAY
    weights = np.exp(-np.log(2) * age_days / half_life_days).astype(np.float32)
    return np.bincount(train_df['item_idx'].to_numpy(np.int32), weights=weights, minlength=n_items).astype(np.float32)

def time_weighted_item_scores(train_df: pd.DataFrame, n_items: int, half_life_days: int) -> np.ndarray:
    return np.log1p(time_weighted_counts(train_df, n_items=n_items, half_life_days=half_life_days)).astype(np.float32)

def trend_score(train_df: pd.DataFrame, n_items: int, short_hl: int, long_hl: int) -> np.ndarray:
    short = time_weighted_counts(train_df, n_items=n_items, half_life_days=short_hl)
    long = time_weighted_counts(train_df, n_items=n_items, half_life_days=long_hl)
    return (np.log1p(short) - np.log1p(long + 1e-06) + 0.15 * np.log1p(short)).astype(np.float32)

@dataclass
class EncodedData:
    train_all: pd.DataFrame
    train_context: pd.DataFrame
    valid: pd.DataFrame
    sample_submission: Optional[pd.DataFrame]
    item_meta: Optional[pd.DataFrame]
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
    out['user_idx'] = out['user_id'].map(user2idx).astype('int32')
    out['item_idx'] = out['item_id'].map(item2idx).astype('int32')
    return out

def split_provided_test(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    valid_keys = test[KEY_COLS].drop_duplicates().assign(_valid_row=1)
    overlap = test[KEY_COLS].drop_duplicates().merge(train[KEY_COLS].drop_duplicates().assign(_in_train=1), on=KEY_COLS, how='left')['_in_train'].fillna(0).mean()
    merged = train.merge(valid_keys, on=KEY_COLS, how='left')
    train_context = merged[merged['_valid_row'].isna()].drop(columns=['_valid_row']).copy()
    valid = test.copy()
    warm_users = set(train_context.user_idx.unique())
    valid = valid[valid.user_idx.isin(warm_users)].copy()
    return (train_context, valid, float(overlap))

def split_temporal_fraction(train: pd.DataFrame, valid_fraction: float=0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = train['timestamp'].quantile(1.0 - valid_fraction)
    train_context = train[train['timestamp'] < cutoff].copy()
    valid = train[train['timestamp'] >= cutoff].copy()
    warm_users = set(train_context.user_idx.unique())
    valid = valid[valid.user_idx.isin(warm_users)].copy()
    return (train_context, valid)

def load_encoded_data(data_dir: Path, validation: str, valid_fraction: float) -> EncodedData:
    train_path = find_csv(data_dir, 'train.csv', required=True)
    test_path = find_csv(data_dir, 'test.csv', required=True)
    sample_path = find_csv(data_dir, 'sample_submission.csv', required=False)
    item_meta_path = find_csv(data_dir, 'item_meta.csv', required=False)
    train_raw = pd.read_csv(train_path).drop_duplicates(KEY_COLS).copy()
    test_raw = pd.read_csv(test_path).drop_duplicates(KEY_COLS).copy()
    sample = pd.read_csv(sample_path) if sample_path is not None else None
    item_meta = pd.read_csv(item_meta_path) if item_meta_path is not None else None
    train_raw['timestamp'] = pd.to_numeric(train_raw['timestamp'], errors='coerce').astype('int64')
    test_raw['timestamp'] = pd.to_numeric(test_raw['timestamp'], errors='coerce').astype('int64')
    all_user_ids = set(train_raw.user_id.unique()) | set(test_raw.user_id.unique())
    if sample is not None and 'user_id' in sample.columns:
        all_user_ids |= set(sample.user_id.unique())
    all_item_ids = set(train_raw.item_id.unique()) | set(test_raw.item_id.unique())
    if item_meta is not None and 'item_id' in item_meta.columns:
        all_item_ids |= set(item_meta.item_id.unique())
    all_user_ids = np.array(sorted((int(u) for u in all_user_ids)))
    all_item_ids = np.array(sorted((int(i) for i in all_item_ids)))
    user2idx = {int(u): i for i, u in enumerate(all_user_ids)}
    item2idx = {int(it): i for i, it in enumerate(all_item_ids)}
    idx2user = {i: int(u) for u, i in user2idx.items()}
    idx2item = {i: int(it) for it, i in item2idx.items()}
    train = add_indices(train_raw, user2idx, item2idx)
    test = add_indices(test_raw, user2idx, item2idx)
    if sample is not None and 'user_id' in sample.columns:
        sample['user_idx'] = sample['user_id'].map(user2idx).astype('int32')
    if validation == 'provided_test':
        train_context, valid, overlap = split_provided_test(train, test)
        if overlap < 0.95:
            print(f'WARNING: provided test overlap is only {overlap:.2%}; using temporal split instead.')
            train_context, valid = split_temporal_fraction(train, valid_fraction=valid_fraction)
            validation_name = f'temporal_fraction_{valid_fraction}'
        else:
            validation_name = 'provided_test_exact_rows_removed'
    elif validation == 'temporal':
        train_context, valid = split_temporal_fraction(train, valid_fraction=valid_fraction)
        validation_name = f'temporal_fraction_{valid_fraction}'
    else:
        raise ValueError(f'Unknown validation mode: {validation}')
    train_item_indices = np.array(sorted(train_context.item_idx.unique()), dtype=np.int32)
    return EncodedData(train_all=train, train_context=train_context, valid=valid, sample_submission=sample, item_meta=item_meta, user2idx=user2idx, item2idx=item2idx, idx2user=idx2user, idx2item=idx2item, n_users=len(all_user_ids), n_items=len(all_item_ids), train_item_indices=train_item_indices, validation_name=validation_name)

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
    deg_inv_sqrt = np.power(deg + 1e-08, -0.5)
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
    return (active_users, pos_arrays, pos_sets)

def sample_bpr_batch(active_users: np.ndarray, pos_arrays: Dict[int, np.ndarray], pos_sets: Dict[int, set], n_items: int, batch_size: int, rng: np.random.Generator):
    users = rng.choice(active_users, size=batch_size, replace=True)
    pos = np.empty(batch_size, dtype=np.int64)
    for r, u in enumerate(users):
        positives = pos_arrays[int(u)]
        pos[r] = positives[rng.integers(0, len(positives))]
    neg = rng.integers(0, n_items, size=batch_size, dtype=np.int64)
    for r, u in enumerate(users):
        seen = pos_sets[int(u)]
        tries = 0
        while int(neg[r]) in seen and tries < 100:
            neg[r] = rng.integers(0, n_items)
            tries += 1
    return (users.astype(np.int64), pos, neg)

@dataclass
class LightGCNConfig:
    dim: int
    layers: int
    epochs: int
    batch_size: int
    lr: float
    reg: float

@dataclass
class Embeddings:
    seed: int
    name: str
    user_emb: np.ndarray
    item_emb: np.ndarray

def train_lightgcn(name: str, seed: int, cfg: LightGCNConfig, X_binary: sparse.csr_matrix, n_users: int, n_items: int, device: torch.device) -> Embeddings:
    seed_everything(seed)
    print(f'\nTraining {name} seed={seed} dim={cfg.dim} layers={cfg.layers} epochs={cfg.epochs}')
    active_users, pos_arrays, pos_sets = prepare_user_positive_arrays(X_binary)
    if len(active_users) == 0:
        raise ValueError(f'{name} has no active users.')
    norm_adj = build_norm_adj(X_binary, n_users, n_items, device)
    model = TorchLightGCN(n_users, n_items, cfg.dim, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, X_binary.nnz // cfg.batch_size)
    for epoch in tqdm(range(1, cfg.epochs + 1), desc=f'{name} seed={seed}'):
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
            reg_loss = cfg.reg * (model.user_emb(users_t).norm(2).pow(2) + model.item_emb(pos_t).norm(2).pow(2) + model.item_emb(neg_t).norm(2).pow(2)) / cfg.batch_size
            loss = mf_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
        if epoch == 1 or epoch % 20 == 0 or epoch == cfg.epochs:
            print(f'{name} seed={seed} epoch {epoch:03d}/{cfg.epochs}, loss={total_loss / steps_per_epoch:.5f}')
    model.eval()
    with torch.no_grad():
        user_e, item_e = model.propagate(norm_adj)
        user_emb = user_e.detach().cpu().numpy().astype(np.float32)
        item_emb = item_e.detach().cpu().numpy().astype(np.float32)
    del model, norm_adj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return Embeddings(seed=seed, name=name, user_emb=user_emb, item_emb=item_emb)

class LightGCNScorer:

    def __init__(self, name: str, emb: Embeddings):
        self.name = name
        self.emb = emb

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.emb.user_emb[user_indices] @ self.emb.item_emb.T).astype(np.float32)

class MultiSeedLightGCNScorer:

    def __init__(self, name: str, embeddings: Sequence[Embeddings], rrf_k: float=60.0):
        self.name = name
        self.embeddings = list(embeddings)
        self.rrf_k = rrf_k

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        out = None
        for emb in self.embeddings:
            raw = emb.user_emb[user_indices] @ emb.item_emb.T
            comp = dense_rrf(raw, rrf_k=self.rrf_k)
            out = comp if out is None else out + comp
        out /= float(len(self.embeddings))
        return out.astype(np.float32)

class VectorScorer:

    def __init__(self, name: str, scores: np.ndarray):
        self.name = name
        self.scores = np.asarray(scores, dtype=np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(user_indices), 1)).astype(np.float32)

class WeightedBlendScorer:

    def __init__(self, name: str, components: Sequence[Tuple[object, float]], norm: str='rrf', rrf_k: float=60.0):
        self.name = name
        self.components = list(components)
        self.norm = norm
        self.rrf_k = rrf_k

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        out = None
        for scorer, weight in self.components:
            s = scorer.score_batch(user_indices)
            s = normalize_dense(s, method=self.norm, rrf_k=self.rrf_k)
            out = weight * s if out is None else out + weight * s
        return out.astype(np.float32)

class DynamicAlphaBlendScorer:

    def __init__(self, name: str, base_scorer, pop_scorer, hist_len: np.ndarray, alpha_bins: Sequence[Tuple[int, float]], norm: str='rrf', rrf_k: float=60.0):
        self.name = name
        self.base_scorer = base_scorer
        self.pop_scorer = pop_scorer
        self.hist_len = hist_len
        self.alpha_bins = list(alpha_bins)
        self.norm = norm
        self.rrf_k = rrf_k

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        base = normalize_dense(self.base_scorer.score_batch(user_indices), method=self.norm, rrf_k=self.rrf_k)
        pop = normalize_dense(self.pop_scorer.score_batch(user_indices), method=self.norm, rrf_k=self.rrf_k)
        alphas = np.array([alpha_for_history_len(int(self.hist_len[int(u)]), self.alpha_bins) for u in user_indices], dtype=np.float32)
        return ((1.0 - alphas[:, None]) * base + alphas[:, None] * pop).astype(np.float32)

class CategoryConditionedPopularityScorer:

    def __init__(self, name: str, train_df: pd.DataFrame, item_meta: pd.DataFrame, item2idx: Dict[int, int], n_items: int, half_life_days: int=180, global_weight: float=0.2):
        self.name = name
        self.n_items = n_items
        self.global_weight = float(global_weight)
        self.item_cat = build_item_category_codes(item_meta, item2idx, n_items)
        self.global_scores = time_weighted_item_scores(train_df, n_items=n_items, half_life_days=half_life_days)
        self.user_dom_cat = self._dominant_user_category(train_df)
        self.cat_scores = {}
        for c in np.unique(self.item_cat):
            if c < 0:
                continue
            s = np.zeros(n_items, dtype=np.float32)
            mask = self.item_cat == c
            s[mask] = self.global_scores[mask]
            self.cat_scores[int(c)] = s

    def _dominant_user_category(self, train_df: pd.DataFrame) -> Dict[int, int]:
        tmp = train_df[['user_idx', 'item_idx']].copy()
        tmp['cat'] = self.item_cat[tmp['item_idx'].to_numpy(np.int32)]
        tmp = tmp[tmp['cat'] >= 0]
        if len(tmp) == 0:
            return {}
        counts = tmp.groupby(['user_idx', 'cat']).size().reset_index(name='cnt')
        counts = counts.sort_values(['user_idx', 'cnt'], ascending=[True, False])
        return counts.drop_duplicates('user_idx').set_index('user_idx')['cat'].astype('int32').to_dict()

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        out = np.empty((len(user_indices), self.n_items), dtype=np.float32)
        for r, u in enumerate(user_indices):
            c = self.user_dom_cat.get(int(u), -1)
            if c in self.cat_scores:
                out[r] = self.global_weight * self.global_scores + self.cat_scores[c]
            else:
                out[r] = self.global_scores
        return out

def build_item_category_codes(item_meta: pd.DataFrame, item2idx: Dict[int, int], n_items: int) -> np.ndarray:
    if item_meta is None or 'item_id' not in item_meta.columns:
        return np.full(n_items, -1, dtype=np.int32)
    meta = item_meta.drop_duplicates('item_id').copy()
    if 'main_category' in meta.columns:
        cat_col = 'main_category'
    elif 'categories' in meta.columns:
        cat_col = 'categories'
    else:
        return np.full(n_items, -1, dtype=np.int32)
    meta = meta[['item_id', cat_col]].copy()
    meta['item_idx'] = meta['item_id'].map(item2idx)
    meta = meta.dropna(subset=['item_idx'])
    meta['item_idx'] = meta['item_idx'].astype('int32')
    meta[cat_col] = meta[cat_col].fillna('').astype(str)
    codes, _ = pd.factorize(meta[cat_col].replace('', np.nan), sort=True)
    meta['cat_code'] = codes.astype('int32')
    item_cat = np.full(n_items, -1, dtype=np.int32)
    valid = meta['cat_code'].to_numpy() >= 0
    item_cat[meta.loc[valid, 'item_idx'].to_numpy(np.int32)] = meta.loc[valid, 'cat_code'].to_numpy(np.int32)
    return item_cat

def build_truth(valid_df: pd.DataFrame) -> Dict[int, set]:
    return valid_df.groupby('user_idx')['item_idx'].apply(lambda x: set((int(v) for v in x.values))).to_dict()

def recall_at_k(scorer, valid_df: pd.DataFrame, X_seen: sparse.csr_matrix, allowed_items: np.ndarray, users: Sequence[int], k: int=10, batch_size: int=64, desc: Optional[str]=None) -> float:
    truth = build_truth(valid_df)
    users = np.array([int(u) for u in users if int(u) in truth], dtype=np.int32)
    if len(users) == 0:
        return float('nan')
    allowed_items = np.asarray(allowed_items, dtype=np.int32)
    recalls = []
    iterator = range(0, len(users), batch_size)
    if desc:
        iterator = tqdm(iterator, desc=desc, leave=False)
    for start in iterator:
        batch_users = users[start:start + batch_size]
        scores = scorer.score_batch(batch_users)
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e+30, posinf=1e+30, neginf=-1e+30)
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
            recalls.append(len(set((int(x) for x in top_items[:k])) & t) / min(k, len(t)))
    return float(np.mean(recalls)) if recalls else float('nan')

def evaluate_and_record(rows: List[dict], name: str, group: str, scorer, data: EncodedData, X_seen: sparse.csr_matrix, all_valid_users: np.ndarray, target_valid_users: Optional[np.ndarray], k: int, batch_size: int, notes: str='') -> dict:
    t0 = time.time()
    recall_all = recall_at_k(scorer, data.valid, X_seen, data.train_item_indices, all_valid_users, k=k, batch_size=batch_size, desc=name)
    recall_target = np.nan
    if target_valid_users is not None and len(target_valid_users) > 0:
        recall_target = recall_at_k(scorer, data.valid, X_seen, data.train_item_indices, target_valid_users, k=k, batch_size=batch_size, desc=f'{name}:target')
    row = {'experiment': name, 'group': group, 'recall_at_10_all_valid_users': recall_all, 'recall_at_10_target_overlap_users': recall_target, 'time_sec': time.time() - t0, 'notes': notes}
    rows.append(row)
    print(f'{name:70s} | all={recall_all:.6f}' + (f' | target={recall_target:.6f}' if not np.isnan(recall_target) else '') + f" | {row['time_sec']:.1f}s")
    return row

def recent_window_df(train_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    cutoff = train_df['timestamp'].max() - window_days * MS_PER_DAY
    out = train_df[train_df['timestamp'] >= cutoff].copy()
    if len(out) == 0:
        raise ValueError(f'No interactions found for recent window {window_days} days.')
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_root', type=str, default='outputs')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--validation', type=str, default='provided_test', choices=['provided_test', 'temporal'])
    parser.add_argument('--valid_fraction', type=float, default=0.15)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', type=float, default=0.0002)
    parser.add_argument('--seeds', type=str, default='42,43,44')
    parser.add_argument('--recent_half_life', type=int, default=180)
    parser.add_argument('--alpha_grid', type=str, default='0.42,0.45,0.475,0.5,0.525,0.55,0.575,0.6')
    parser.add_argument('--blend_norm', type=str, default='rrf', choices=['rrf', 'zscore', 'minmax', 'none'])
    parser.add_argument('--rrf_k', type=float, default=60.0)
    parser.add_argument('--dynamic_alpha_specs', type=str, default='3:0.65,6:0.55,12:0.50,999999:0.45;3:0.75,6:0.65,12:0.55,999999:0.45;3:0.60,6:0.55,12:0.50,999999:0.40')
    parser.add_argument('--recent_lgcn_windows', type=str, default='180,365')
    parser.add_argument('--recent_lgcn_epochs', type=int, default=160)
    parser.add_argument('--recent_lgcn_seeds', type=str, default='42')
    parser.add_argument('--trend_specs', type=str, default='30:365,60:365,90:365,120:365')
    parser.add_argument('--extra_weight_grid', type=str, default='0.05,0.1,0.15,0.2,0.3')
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--max_eval_users', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    if args.quick:
        print('Running in --quick mode.')
        args.dim = min(args.dim, 128)
        args.epochs = min(args.epochs, 30)
        args.recent_lgcn_epochs = min(args.recent_lgcn_epochs, 20)
        args.seeds = args.seeds.split(',')[0]
        args.recent_lgcn_windows = args.recent_lgcn_windows.split(',')[0]
        args.max_eval_users = args.max_eval_users or 1000
    seed_everything(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    run_id = args.run_id or now_run_id()
    out_dir = Path(args.output_root) / f'lightgcn_feature_additions_{run_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print('Output dir:', out_dir.resolve())
    print('Device:', device)
    data = load_encoded_data(Path(args.data_dir), validation=args.validation, valid_fraction=args.valid_fraction)
    X_context = make_interaction_matrix(data.train_context, data.n_users, data.n_items, binary_after_sum=True)
    all_valid_users = np.array(sorted(data.valid.user_idx.unique()), dtype=np.int32)
    if args.max_eval_users is not None and len(all_valid_users) > args.max_eval_users:
        rng = np.random.default_rng(args.seed)
        all_valid_users = np.array(sorted(rng.choice(all_valid_users, size=args.max_eval_users, replace=False)), dtype=np.int32)
    target_valid_users = None
    if data.sample_submission is not None and 'user_idx' in data.sample_submission.columns:
        target_valid_users = np.array(sorted(set(all_valid_users) & set(data.sample_submission.user_idx.astype(int))), dtype=np.int32)
    eda = {'validation_name': data.validation_name, 'n_users': data.n_users, 'n_items': data.n_items, 'train_all_rows': int(len(data.train_all)), 'train_context_rows': int(len(data.train_context)), 'valid_rows': int(len(data.valid)), 'valid_users': int(data.valid.user_idx.nunique()), 'eval_users': int(len(all_valid_users)), 'target_overlap_eval_users': int(len(target_valid_users)) if target_valid_users is not None else 0, 'train_context_items': int(len(data.train_item_indices))}
    with open(out_dir / 'eda.json', 'w') as f:
        json.dump(eda, f, indent=2)
    print(json.dumps(eda, indent=2))
    rows: List[dict] = []
    alpha_grid = parse_float_list(args.alpha_grid)
    extra_weight_grid = parse_float_list(args.extra_weight_grid)
    seeds = parse_int_list(args.seeds)
    recent_lgcn_seeds = parse_int_list(args.recent_lgcn_seeds)
    recent_windows = parse_int_list(args.recent_lgcn_windows)
    dynamic_specs = parse_dynamic_alpha_specs(args.dynamic_alpha_specs)
    trend_specs = parse_trend_specs(args.trend_specs)
    cfg = LightGCNConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)
    full_embeddings: List[Embeddings] = []
    for seed in seeds:
        emb = train_lightgcn(name='LightGCN_full', seed=seed, cfg=cfg, X_binary=X_context, n_users=data.n_users, n_items=data.n_items, device=device)
        full_embeddings.append(emb)
    full_single = LightGCNScorer('LightGCN_full_seed' + str(seeds[0]), full_embeddings[0])
    full_multi = MultiSeedLightGCNScorer('LightGCN_full_multiseed_' + '-'.join(map(str, seeds)), full_embeddings, rrf_k=args.rrf_k)
    recent_pop = VectorScorer(f'recent_pop_{args.recent_half_life}d', time_weighted_item_scores(data.train_context, data.n_items, args.recent_half_life))
    hist_len = np.diff(X_context.indptr)
    evaluate_and_record(rows, full_single.name, 'baseline', full_single, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Single-seed full LightGCN.')
    evaluate_and_record(rows, recent_pop.name, 'standalone_signal', recent_pop, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Recent popularity standalone.')
    for alpha in alpha_grid:
        scorer = WeightedBlendScorer(f'single_LightGCN + recent{args.recent_half_life} alpha={alpha:g}', [(full_single, 1.0 - alpha), (recent_pop, alpha)], norm=args.blend_norm, rrf_k=args.rrf_k)
        evaluate_and_record(rows, scorer.name, 'current_family_single_seed', scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Fixed-alpha current best family.')
    for spec in dynamic_specs:
        scorer = DynamicAlphaBlendScorer(f'dynamic_alpha_single_{spec_to_name(spec)}', full_single, recent_pop, hist_len=hist_len, alpha_bins=spec, norm=args.blend_norm, rrf_k=args.rrf_k)
        evaluate_and_record(rows, scorer.name, 'addition_dynamic_alpha', scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'User-history-length dependent alpha.')
    if len(full_embeddings) > 1:
        evaluate_and_record(rows, full_multi.name, 'addition_multiseed', full_multi, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'RRF-average of multiple LightGCN seeds.')
        for alpha in alpha_grid:
            scorer = WeightedBlendScorer(f'multiseed_LightGCN + recent{args.recent_half_life} alpha={alpha:g}', [(full_multi, 1.0 - alpha), (recent_pop, alpha)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_multiseed', scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed LightGCN plus recent popularity.')
        for spec in dynamic_specs:
            scorer = DynamicAlphaBlendScorer(f'dynamic_alpha_multiseed_{spec_to_name(spec)}', full_multi, recent_pop, hist_len=hist_len, alpha_bins=spec, norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_dynamic_alpha_multiseed', scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Dynamic alpha using multi-seed LightGCN.')
    recent_cfg = LightGCNConfig(dim=args.dim, layers=args.layers, epochs=args.recent_lgcn_epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)
    for window in recent_windows:
        recent_df = recent_window_df(data.train_context, window)
        X_recent = make_interaction_matrix(recent_df, data.n_users, data.n_items, binary_after_sum=True)
        recent_embeddings = []
        for seed in recent_lgcn_seeds:
            emb = train_lightgcn(name=f'LightGCN_recent{window}d', seed=seed, cfg=recent_cfg, X_binary=X_recent, n_users=data.n_users, n_items=data.n_items, device=device)
            recent_embeddings.append(emb)
        if len(recent_embeddings) == 1:
            recent_lgcn = LightGCNScorer(f'LightGCN_recent{window}d_seed{recent_embeddings[0].seed}', recent_embeddings[0])
        else:
            recent_lgcn = MultiSeedLightGCNScorer(f'LightGCN_recent{window}d_multiseed_' + '-'.join(map(str, recent_lgcn_seeds)), recent_embeddings, rrf_k=args.rrf_k)
        evaluate_and_record(rows, recent_lgcn.name, 'addition_recent_window_lgcn', recent_lgcn, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, f'LightGCN trained only on last {window} days.')
        for gamma in extra_weight_grid:
            scorer = WeightedBlendScorer(f'single_LightGCN + recent_pop + recentLGCN{window} gamma={gamma:g}', [(full_single, 0.5), (recent_pop, 0.5), (recent_lgcn, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_recent_window_lgcn', scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Base model plus recent-window LightGCN.')
            if len(full_embeddings) > 1:
                scorer2 = WeightedBlendScorer(f'multiseed_LightGCN + recent_pop + recentLGCN{window} gamma={gamma:g}', [(full_multi, 0.5), (recent_pop, 0.5), (recent_lgcn, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
                evaluate_and_record(rows, scorer2.name, 'addition_recent_window_lgcn_multiseed', scorer2, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed current best plus recent-window LightGCN.')
        del X_recent, recent_embeddings
        gc.collect()
    for short_hl, long_hl in trend_specs:
        trend = VectorScorer(f'trend_{short_hl}v{long_hl}', trend_score(data.train_context, data.n_items, short_hl=short_hl, long_hl=long_hl))
        evaluate_and_record(rows, trend.name, 'addition_trend_standalone', trend, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Relative trend / acceleration standalone.')
        for gamma in extra_weight_grid:
            scorer = WeightedBlendScorer(f'single_LightGCN + recent_pop + trend{short_hl}v{long_hl} gamma={gamma:g}', [(full_single, 0.5), (recent_pop, 0.5), (trend, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_trend', scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Base model plus trend acceleration.')
            if len(full_embeddings) > 1:
                scorer2 = WeightedBlendScorer(f'multiseed_LightGCN + recent_pop + trend{short_hl}v{long_hl} gamma={gamma:g}', [(full_multi, 0.5), (recent_pop, 0.5), (trend, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
                evaluate_and_record(rows, scorer2.name, 'addition_trend_multiseed', scorer2, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed current best plus trend acceleration.')
    if data.item_meta is not None:
        cat_pop = CategoryConditionedPopularityScorer(f'category_conditioned_recent{args.recent_half_life}', data.train_context, data.item_meta, data.item2idx, n_items=data.n_items, half_life_days=args.recent_half_life, global_weight=0.2)
        evaluate_and_record(rows, cat_pop.name, 'addition_category_conditioned_standalone', cat_pop, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'User dominant-category conditioned recent popularity.')
        for gamma in extra_weight_grid:
            scorer = WeightedBlendScorer(f'single_LightGCN + recent_pop + cat_recent gamma={gamma:g}', [(full_single, 0.5), (recent_pop, 0.5), (cat_pop, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_category_conditioned', scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Base model plus user-category-conditioned recent popularity.')
            if len(full_embeddings) > 1:
                scorer2 = WeightedBlendScorer(f'multiseed_LightGCN + recent_pop + cat_recent gamma={gamma:g}', [(full_multi, 0.5), (recent_pop, 0.5), (cat_pop, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
                evaluate_and_record(rows, scorer2.name, 'addition_category_conditioned_multiseed', scorer2, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed current best plus category-conditioned recent popularity.')
    else:
        print('No item_meta.csv found; skipping category-conditioned recent popularity.')
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / 'results.csv', index=False)
    sorted_results = results.sort_values(['recall_at_10_all_valid_users', 'recall_at_10_target_overlap_users'], ascending=[False, False], na_position='last')
    sorted_results.to_csv(out_dir / 'results_sorted.csv', index=False)
    print('\n=== Top by all-valid Recall@10 ===')
    print(sorted_results.head(25).to_string(index=False))
    sorted_target = results.sort_values(['recall_at_10_target_overlap_users', 'recall_at_10_all_valid_users'], ascending=[False, False], na_position='last')
    sorted_target.to_csv(out_dir / 'results_sorted_target.csv', index=False)
    print('\n=== Top by target-overlap Recall@10 ===')
    print(sorted_target.head(25).to_string(index=False))
    print('\nSaved:')
    print(' ', out_dir / 'results.csv')
    print(' ', out_dir / 'results_sorted.csv')
    print(' ', out_dir / 'results_sorted_target.csv')
    print(' ', out_dir / 'config.json')
    print(' ', out_dir / 'eda.json')
if __name__ == '__main__':
    main()
