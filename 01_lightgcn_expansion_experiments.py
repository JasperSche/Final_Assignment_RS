#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, random, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
MS_PER_DAY = 1000 * 60 * 60 * 24
KEY_COLS = ['user_id', 'item_id', 'timestamp']

# Small utilities kept local so this script can run as a standalone experiment.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_csv(data_dir: Path, canonical: str) -> Path:
    p = data_dir / canonical
    if p.exists():
        return p
    stem = canonical.replace('.csv', '')
    matches = sorted(data_dir.glob(f'{stem}*.csv'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'Could not find {canonical} or {stem}*.csv under {data_dir}')

def now_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

@dataclass
class EncodedData:
    train_all: pd.DataFrame
    train_context: pd.DataFrame
    valid: pd.DataFrame
    item_meta_raw: pd.DataFrame
    sample_submission: Optional[pd.DataFrame]
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    idx2user: Dict[int, int]
    idx2item: Dict[int, int]
    n_users: int
    n_items: int
    train_item_indices: np.ndarray
    item_static: pd.DataFrame
    validation_name: str

def coerce_text(x) -> str:
    return '' if pd.isna(x) else str(x)

def build_item_static(item_meta: pd.DataFrame, all_item_ids: Sequence[int], item2idx: Dict[int, int]) -> pd.DataFrame:
    # Keep one dense row per encoded item; missing metadata is allowed.
    meta = item_meta.drop_duplicates('item_id').copy()
    base = pd.DataFrame({'item_id': all_item_ids})
    base['item_idx'] = base['item_id'].map(item2idx).astype('int32')
    base = base.merge(meta, on='item_id', how='left')
    text_cols = ['main_category', 'title', 'features', 'description', 'store', 'categories', 'details', 'subtitle']
    for c in text_cols:
        if c not in base:
            base[c] = ''
        base[c] = base[c].map(coerce_text)
    for c in ['main_category', 'store']:
        values = base[c].replace('', np.nan)
        codes, _ = pd.factorize(values, sort=True)
        base[f'{c}_code'] = codes.astype('int32')
    for c in ['average_rating', 'rating_number', 'price']:
        if c not in base:
            base[c] = np.nan
        base[c] = pd.to_numeric(base[c], errors='coerce')
    base['has_metadata'] = base['title'].ne('').astype('int8')
    base['text_for_content'] = (base['title'] + ' ' + base['main_category'] + ' ' + base['categories'] + ' ' + base['features'] + ' ' + base['description'] + ' ' + base['store'] + ' ' + base['details']).str.replace('\\s+', ' ', regex=True).str.strip()
    keep = ['item_idx', 'item_id', 'main_category_code', 'store_code', 'has_metadata', 'average_rating', 'rating_number', 'price', 'text_for_content']
    return base[keep].sort_values('item_idx').reset_index(drop=True)

def add_indices(df: pd.DataFrame, user2idx: Dict[int, int], item2idx: Dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out['user_idx'] = out['user_id'].map(user2idx).astype('int32')
    out['item_idx'] = out['item_id'].map(item2idx).astype('int32')
    return out

def split_provided_test(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    # The provided test is only useful as validation when its rows also exist in train.
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
    return (train_context, valid[valid.user_idx.isin(warm_users)].copy())

def load_encoded_data(data_dir: Path, validation: str, valid_fraction: float) -> EncodedData:
    train_raw = pd.read_csv(find_csv(data_dir, 'train.csv')).drop_duplicates(KEY_COLS).copy()
    test_raw = pd.read_csv(find_csv(data_dir, 'test.csv')).drop_duplicates(KEY_COLS).copy()
    item_meta_raw = pd.read_csv(find_csv(data_dir, 'item_meta.csv'))
    sample_submission = None
    try:
        sample_submission = pd.read_csv(find_csv(data_dir, 'sample_submission.csv'))
    except FileNotFoundError:
        pass
    train_raw['timestamp'] = pd.to_numeric(train_raw['timestamp'], errors='coerce').astype('int64')
    test_raw['timestamp'] = pd.to_numeric(test_raw['timestamp'], errors='coerce').astype('int64')
    all_user_ids = set(train_raw.user_id.unique()) | set(test_raw.user_id.unique())
    if sample_submission is not None and 'user_id' in sample_submission.columns:
        all_user_ids |= set(sample_submission.user_id.unique())
    all_item_ids = set(train_raw.item_id.unique()) | set(test_raw.item_id.unique()) | set(item_meta_raw.item_id.unique())
    all_user_ids = np.array(sorted(all_user_ids))
    all_item_ids = np.array(sorted(all_item_ids))
    user2idx = {int(u): i for i, u in enumerate(all_user_ids)}
    item2idx = {int(it): i for i, it in enumerate(all_item_ids)}
    idx2user = {i: int(u) for u, i in user2idx.items()}
    idx2item = {i: int(it) for it, i in item2idx.items()}
    train = add_indices(train_raw, user2idx, item2idx)
    test = add_indices(test_raw, user2idx, item2idx)
    if sample_submission is not None and 'user_id' in sample_submission.columns:
        sample_submission['user_idx'] = sample_submission['user_id'].map(user2idx).astype('int32')
    if validation == 'provided_test':
        train_context, valid, overlap = split_provided_test(train, test)
        if overlap < 0.95:
            print(f'WARNING: only {overlap:.1%} of test rows are exact train rows. Falling back to temporal split.')
            train_context, valid = split_temporal_fraction(train, valid_fraction)
            validation_name = f'temporal_fraction_{valid_fraction}'
        else:
            validation_name = 'provided_test_exact_rows_removed'
    elif validation == 'temporal':
        train_context, valid = split_temporal_fraction(train, valid_fraction)
        validation_name = f'temporal_fraction_{valid_fraction}'
    else:
        raise ValueError(validation)
    train_item_indices = np.array(sorted(train_context.item_idx.unique()), dtype=np.int32)
    item_static = build_item_static(item_meta_raw, all_item_ids, item2idx)
    return EncodedData(train, train_context, valid, item_meta_raw, sample_submission, user2idx, item2idx, idx2user, idx2item, len(all_user_ids), len(all_item_ids), train_item_indices, item_static, validation_name)

def make_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int, weights: Optional[np.ndarray]=None, binary_after_sum: bool=False) -> sparse.csr_matrix:
    rows = df['user_idx'].to_numpy(np.int32)
    cols = df['item_idx'].to_numpy(np.int32)
    vals = np.ones(len(df), dtype=np.float32) if weights is None else weights.astype(np.float32)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    X.sum_duplicates()
    if binary_after_sum:
        X.data[:] = 1.0
    return X

class TorchLightGCN(nn.Module):

    def __init__(self, n_users: int, n_items: int, dim: int=128, n_layers: int=3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = (n_users, n_items, n_layers)
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        final = torch.stack(embs, dim=0).mean(dim=0)
        return torch.split(final, [self.n_users, self.n_items], dim=0)

def build_norm_adj(X_weighted: sparse.csr_matrix, n_users: int, n_items: int, device: torch.device) -> torch.Tensor:
    # Build the symmetric LightGCN graph and apply degree normalization.
    X = X_weighted.tocoo()
    u = X.row.astype(np.int64)
    i = X.col.astype(np.int64) + n_users
    w = X.data.astype(np.float32)
    rows = np.concatenate([u, i])
    cols = np.concatenate([i, u])
    vals_raw = np.concatenate([w, w])
    deg = np.bincount(rows, weights=vals_raw, minlength=n_users + n_items).astype(np.float32)
    vals = vals_raw * np.power(deg[rows] + 1e-08, -0.5) * np.power(deg[cols] + 1e-08, -0.5)
    idx = torch.LongTensor(np.vstack([rows, cols]))
    val = torch.FloatTensor(vals)
    return torch.sparse_coo_tensor(idx, val, size=(n_users + n_items, n_users + n_items)).coalesce().to(device)

def prepare_user_positive_arrays(X_binary: sparse.csr_matrix):
    X = X_binary.tocsr()
    active_users = np.where(np.diff(X.indptr) > 0)[0].astype(np.int32)
    pos_arrays, pos_sets = ({}, {})
    for u in active_users:
        arr = X[u].indices.astype(np.int32)
        pos_arrays[int(u)] = arr
        pos_sets[int(u)] = set(arr.tolist())
    return (active_users, pos_arrays, pos_sets)

def build_popularity_sampler(X_binary: sparse.csr_matrix, power: float=0.75) -> np.ndarray:
    counts = np.asarray(X_binary.sum(axis=0)).ravel().astype(np.float64)
    probs = np.power(counts + 1e-06, power)
    return probs / probs.sum()

def sample_bpr_batch(active_users, pos_arrays, pos_sets, n_items, batch_size, rng, neg_probs=None):
    # Draw one observed item and one unseen negative item for each sampled user.
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
            neg[r] = rng.integers(0, n_items) if neg_probs is None else rng.choice(np.arange(n_items), p=neg_probs)
            tries += 1
    return (users.astype(np.int64), pos, neg)

@dataclass
class LightGCNConfig:
    name: str
    dim: int = 128
    layers: int = 3
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 0.002
    reg: float = 0.0001
    seed: int = 42
    neg_sampling: str = 'uniform'

class LightGCNScorer:

    def __init__(self, name: str, user_emb: np.ndarray, item_emb: np.ndarray):
        self.name = name
        self.user_emb = user_emb.astype(np.float32)
        self.item_emb = item_emb.astype(np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_emb[user_indices] @ self.item_emb.T

def train_lightgcn(cfg: LightGCNConfig, X_binary: sparse.csr_matrix, X_graph: sparse.csr_matrix, n_users: int, n_items: int, device: torch.device) -> LightGCNScorer:
    seed_everything(cfg.seed)
    print(f'\nTraining {cfg.name} on {device} | dim={cfg.dim}, layers={cfg.layers}, epochs={cfg.epochs}, neg={cfg.neg_sampling}')
    active_users, pos_arrays, pos_sets = prepare_user_positive_arrays(X_binary)
    neg_probs = build_popularity_sampler(X_binary) if cfg.neg_sampling == 'popularity' else None
    norm_adj = build_norm_adj(X_graph, n_users, n_items, device)
    model = TorchLightGCN(n_users, n_items, cfg.dim, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    steps_per_epoch = max(1, X_binary.nnz // cfg.batch_size)
    for epoch in tqdm(range(1, cfg.epochs + 1), desc=cfg.name):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            users, pos, neg = sample_bpr_batch(active_users, pos_arrays, pos_sets, n_items, cfg.batch_size, rng, neg_probs)
            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos).to(device)
            neg_t = torch.LongTensor(neg).to(device)
            # Recompute propagated embeddings after each update
            user_e, item_e = model.propagate(norm_adj)
            u_e, p_e, n_e = (user_e[users_t], item_e[pos_t], item_e[neg_t])
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
            print(f'{cfg.name} epoch {epoch:03d}/{cfg.epochs}, loss={total_loss / steps_per_epoch:.5f}')
    model.eval()
    with torch.no_grad():
        user_e, item_e = model.propagate(norm_adj)
        user_emb = user_e.detach().cpu().numpy().astype(np.float32)
        item_emb = item_e.detach().cpu().numpy().astype(np.float32)
    del model, norm_adj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return LightGCNScorer(cfg.name, user_emb, item_emb)

class VectorScorer:

    def __init__(self, name: str, item_scores: np.ndarray):
        self.name = name
        self.item_scores = item_scores.astype(np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores, (len(user_indices), 1))

class SparseLinearScorer:

    def __init__(self, name: str, X_user_item: sparse.csr_matrix, W_item_item: sparse.csr_matrix):
        self.name = name
        self.X = X_user_item.tocsr()
        self.W = W_item_item.tocsr()

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        S = self.X[user_indices] @ self.W
        return S.toarray().astype(np.float32) if sparse.issparse(S) else S.astype(np.float32)

class CategoryPopularityScorer:

    def __init__(self, name: str, train_df: pd.DataFrame, item_static: pd.DataFrame, n_items: int, half_life_days: int=365):
        self.name = name
        self.n_items = n_items
        self.item_cat = item_static.set_index('item_idx')['main_category_code'].reindex(range(n_items)).fillna(-1).astype('int32').to_numpy()
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

    def _dominant_user_category(self, train_df):
        tmp = train_df[['user_idx', 'item_idx']].copy()
        tmp['cat'] = self.item_cat[tmp['item_idx'].values]
        tmp = tmp[tmp['cat'] >= 0]
        if len(tmp) == 0:
            return {}
        counts = tmp.groupby(['user_idx', 'cat']).size().reset_index(name='cnt').sort_values(['user_idx', 'cnt'], ascending=[True, False])
        return counts.drop_duplicates('user_idx').set_index('user_idx')['cat'].astype('int32').to_dict()

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        out = np.empty((len(user_indices), self.n_items), dtype=np.float32)
        for r, u in enumerate(user_indices):
            c = self.user_dom_cat.get(int(u), -1)
            out[r] = 0.15 * self.global_scores + self.cat_scores[c] if c in self.cat_scores else self.global_scores
        return out

class ContentTfidfScorer:

    def __init__(self, name: str, X_user_item: sparse.csr_matrix, item_static: pd.DataFrame, n_items: int, quick: bool=False):
        self.name = name
        self.X = X_user_item.tocsr()
        self.n_items = n_items
        text = item_static.sort_values('item_idx')['text_for_content'].fillna('').astype(str).values
        max_word = 80000 if quick else 200000
        max_char = 40000 if quick else 100000
        # Word and character features make the content signal less brittle to sparse titles.
        self.vectorizer = FeatureUnion([('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True, dtype=np.float32)), ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=max_char, sublinear_tf=True, dtype=np.float32))])
        print('Fitting metadata TF-IDF content scorer...')
        self.F_items = normalize(self.vectorizer.fit_transform(text), norm='l2', axis=1).tocsr().astype(np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        P = normalize(self.X[user_indices] @ self.F_items, norm='l2', axis=1)
        S = P @ self.F_items.T
        return S.toarray().astype(np.float32) if sparse.issparse(S) else S.astype(np.float32)

def sanitize_scores(scores: np.ndarray, bad_value: float=0.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    return np.nan_to_num(scores, nan=bad_value, posinf=1000000.0, neginf=-1000000.0).astype(np.float32, copy=False)

def normalize_score_matrix(scores: np.ndarray, method: str='zscore') -> np.ndarray:
    scores = sanitize_scores(scores, bad_value=0.0)
    if method == 'none':
        return scores
    if method == 'zscore':
        mu = scores.mean(axis=1, keepdims=True)
        sigma = scores.std(axis=1, keepdims=True)
        return ((scores - mu) / (sigma + 1e-06)).astype(np.float32)
    if method == 'minmax':
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
        return ((scores - lo) / (hi - lo + 1e-06)).astype(np.float32)
    if method == 'rrf':
        order = np.argsort(-scores, axis=1)
        ranks = np.empty_like(order, dtype=np.int32)
        row_idx = np.arange(scores.shape[0])[:, None]
        ranks[row_idx, order] = np.arange(scores.shape[1], dtype=np.int32)[None, :]
        return (1.0 / (60.0 + ranks + 1.0)).astype(np.float32)
    raise ValueError(method)

class BlendScorer:

    def __init__(self, name: str, base, addon, alpha: float, norm: str='zscore'):
        self.name = name
        self.base = base
        self.addon = addon
        self.alpha = float(alpha)
        self.norm = norm

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        a = normalize_score_matrix(self.base.score_batch(user_indices), self.norm)
        b = normalize_score_matrix(self.addon.score_batch(user_indices), self.norm)
        return ((1.0 - self.alpha) * a + self.alpha * b).astype(np.float32)

class WeightedBlendScorer:

    def __init__(self, name: str, components: List[Tuple[object, float]], norm: str='zscore'):
        self.name = name
        self.components = components
        self.norm = norm

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        out = None
        for scorer, weight in self.components:
            s = normalize_score_matrix(scorer.score_batch(user_indices), self.norm)
            out = weight * s if out is None else out + weight * s
        return out.astype(np.float32)

def time_decay_weights(train_df: pd.DataFrame, half_life_days: int) -> np.ndarray:
    max_ts = train_df['timestamp'].max()
    age_days = (max_ts - train_df['timestamp'].to_numpy()) / MS_PER_DAY
    return np.exp(-np.log(2) * age_days / half_life_days).astype(np.float32)

def time_weighted_item_scores(train_df: pd.DataFrame, n_items: int, half_life_days: Optional[int]=None, since_days: Optional[int]=None) -> np.ndarray:
    df = train_df
    if since_days is not None:
        df = df[df['timestamp'] >= train_df['timestamp'].max() - since_days * MS_PER_DAY]
    if len(df) == 0:
        return np.zeros(n_items, dtype=np.float32)
    if half_life_days is None:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        age_days = (train_df['timestamp'].max() - df['timestamp'].to_numpy()) / MS_PER_DAY
        weights = np.exp(-np.log(2) * age_days / half_life_days).astype(np.float32)
    scores = np.bincount(df['item_idx'].to_numpy(np.int32), weights=weights, minlength=n_items).astype(np.float32)
    return np.log1p(scores).astype(np.float32)

def topk_sparse_rows(M: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    M = M.tocsr()
    rows = []
    cols = []
    data = []
    for i in tqdm(range(M.shape[0]), desc=f'topk rows k={k}', leave=False):
        start, end = (M.indptr[i], M.indptr[i + 1])
        c = M.indices[start:end]
        d = M.data[start:end]
        if len(d) == 0:
            continue
        if len(d) > k:
            idx = np.argpartition(-d, k - 1)[:k]
            c = c[idx]
            d = d[idx]
        order = np.argsort(-d)
        c = c[order]
        d = d[order]
        rows.extend([i] * len(c))
        cols.extend(c.tolist())
        data.extend(d.astype(np.float32).tolist())
    out = sparse.csr_matrix((data, (rows, cols)), shape=M.shape, dtype=np.float32)
    out.eliminate_zeros()
    return out

def build_itemknn_cosine(X: sparse.csr_matrix, topk: int=300) -> sparse.csr_matrix:
    print(f'Building itemKNN cosine topk={topk}')
    Y = normalize(X.T, norm='l2', axis=1)
    W = Y @ Y.T
    W = W.tolil()
    W.setdiag(0)
    W = W.tocsr()
    W.eliminate_zeros()
    return topk_sparse_rows(W, topk)

def build_sequence_transition_matrix(train_df: pd.DataFrame, n_items: int, topk: int=300, max_gap_days: Optional[int]=None) -> sparse.csr_matrix:
    print(f'Building sequence transition matrix topk={topk}')
    rows = []
    cols = []
    data = []
    for _, g in tqdm(train_df.sort_values(['user_idx', 'timestamp']).groupby('user_idx'), desc='user sequences', leave=False):
        items = g['item_idx'].to_numpy(np.int32)
        ts = g['timestamp'].to_numpy(np.int64)
        if len(items) < 2:
            continue
        # Count adjacent interactions as directional transitions.
        for a in range(len(items) - 1):
            i, j = (int(items[a]), int(items[a + 1]))
            if i == j:
                continue
            if max_gap_days is not None:
                gap_days = (ts[a + 1] - ts[a]) / MS_PER_DAY
                if gap_days < 0 or gap_days > max_gap_days:
                    continue
            rows.append(i)
            cols.append(j)
            data.append(0.5 + (a + 1) / max(1, len(items) - 1))
    if not rows:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    W = sparse.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    W.sum_duplicates()
    W = normalize(W, norm='l1', axis=1)
    return topk_sparse_rows(W, topk)

def build_recent_history_matrix(train_df: pd.DataFrame, n_users: int, n_items: int, max_len: int=5) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []
    for u, g in train_df.sort_values(['user_idx', 'timestamp']).groupby('user_idx'):
        items = g['item_idx'].to_numpy(np.int32)[-max_len:]
        if len(items) == 0:
            continue
        weights = np.linspace(0.5, 1.0, len(items), dtype=np.float32)
        rows.extend([int(u)] * len(items))
        cols.extend(items.tolist())
        data.extend(weights.tolist())
    H = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    H.sum_duplicates()
    return H

class SASRecDataset(Dataset):

    def __init__(self, train_df: pd.DataFrame, max_len: int, n_items: int, seed: int=42, max_examples_per_user: Optional[int]=None):
        self.max_len = max_len
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)
        self.seqs = train_df.sort_values(['user_idx', 'timestamp']).groupby('user_idx')['item_idx'].apply(lambda x: x.to_numpy(np.int32)).to_dict()
        self.user_seen = {int(u): set(arr.tolist()) for u, arr in self.seqs.items()}
        self.examples = []
        for u, arr in self.seqs.items():
            if len(arr) < 2:
                continue
            positions = list(range(1, len(arr)))
            if max_examples_per_user is not None and len(positions) > max_examples_per_user:
                positions = positions[-max_examples_per_user:]
            for t in positions:
                self.examples.append((int(u), int(t)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        u, t = self.examples[idx]
        arr = self.seqs[u]
        prefix = arr[max(0, t - self.max_len):t]
        target = int(arr[t])
        seq = np.zeros(self.max_len, dtype=np.int64)
        seq[-len(prefix):] = prefix + 1
        neg = self.rng.integers(0, self.n_items)
        seen = self.user_seen[u]
        tries = 0
        while int(neg) in seen and tries < 100:
            neg = self.rng.integers(0, self.n_items)
            tries += 1
        return (torch.LongTensor(seq), torch.LongTensor([target + 1]).squeeze(0), torch.LongTensor([int(neg) + 1]).squeeze(0))

class SASRecNet(nn.Module):

    def __init__(self, n_items: int, max_len: int=30, hidden_dim: int=128, n_layers: int=2, n_heads: int=2, dropout: float=0.25):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(positions))
        causal_mask = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()
        padding_mask = seq.eq(0)
        lengths_raw = seq.ne(0).sum(dim=1)
        h = torch.zeros((B, self.item_emb.embedding_dim), device=seq.device, dtype=x.dtype)
        valid_rows = lengths_raw > 0
        if valid_rows.any():
            # Use the last real position as the user's current sequential state.
            out = self.encoder(x[valid_rows], mask=causal_mask, src_key_padding_mask=padding_mask[valid_rows])
            out = self.norm(out)
            lengths = lengths_raw[valid_rows].clamp(min=1) - 1
            h[valid_rows] = out[torch.arange(out.shape[0], device=seq.device), lengths]
        return torch.nan_to_num(h, nan=0.0, posinf=1000000.0, neginf=-1000000.0)

class SASRecScorer:

    def __init__(self, name: str, model: SASRecNet, train_df: pd.DataFrame, n_items: int, max_len: int, device: torch.device):
        self.name = name
        self.model = model.eval()
        self.n_items = n_items
        self.max_len = max_len
        self.device = device
        self.user_sequences = train_df.sort_values(['user_idx', 'timestamp']).groupby('user_idx')['item_idx'].apply(lambda x: x.to_numpy(np.int32)).to_dict()

    def _make_seq_batch(self, user_indices):
        seqs = np.zeros((len(user_indices), self.max_len), dtype=np.int64)
        for r, u in enumerate(user_indices):
            hist = self.user_sequences.get(int(u), np.array([], dtype=np.int32))[-self.max_len:]
            if len(hist):
                seqs[r, -len(hist):] = hist + 1
        return seqs

    def score_batch(self, user_indices):
        outs = []
        bs = 512
        with torch.no_grad():
            for st in range(0, len(user_indices), bs):
                batch = user_indices[st:st + bs]
                seq = torch.LongTensor(self._make_seq_batch(batch)).to(self.device)
                h = self.model(seq)
                scores = h @ self.model.item_emb.weight[1:].T
                scores = torch.nan_to_num(scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
                outs.append(scores.detach().cpu().numpy().astype(np.float32))
        return sanitize_scores(np.vstack(outs), bad_value=0.0)

def train_sasrec(name: str, train_df: pd.DataFrame, n_items: int, device: torch.device, epochs: int, hidden_dim: int, layers: int, heads: int, max_len: int, dropout: float, batch_size: int, seed: int):
    seed_everything(seed)
    print(f'\nTraining {name} on {device} | hidden={hidden_dim}, layers={layers}, epochs={epochs}')
    dataset = SASRecDataset(train_df, max_len=max_len, n_items=n_items, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    model = SASRecNet(n_items, max_len, hidden_dim, layers, heads, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-05)
    for epoch in tqdm(range(1, epochs + 1), desc=name):
        model.train()
        total = 0.0
        for seq, pos, neg in loader:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            h = model(seq)
            pos_e = model.item_emb(pos)
            neg_e = model.item_emb(neg)
            loss = -F.logsigmoid((h * pos_e).sum(dim=1) - (h * neg_e).sum(dim=1)).mean()
            if not torch.isfinite(loss):
                print('WARNING: non-finite SASRec loss encountered; skipping batch')
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu())
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'{name} epoch {epoch:03d}/{epochs}, loss={total / max(1, len(loader)):.5f}')
    return SASRecScorer(name, model, train_df, n_items, max_len, device)

def build_truth(valid_df):
    return valid_df.groupby('user_idx')['item_idx'].apply(lambda x: set((int(v) for v in x.values))).to_dict()

def estimate_recall_upper_bound(valid_df, X_seen, allowed_items, users, k):
    allowed = set((int(x) for x in allowed_items))
    truth = build_truth(valid_df)
    vals = []
    for u in users:
        t = truth.get(int(u), set())
        if not t:
            continue
        seen = set((int(x) for x in X_seen[int(u)].indices))
        reachable = [it for it in t if it in allowed and it not in seen]
        vals.append(min(len(reachable), k) / min(len(t), k))
    return float(np.mean(vals)) if vals else float('nan')

def recall_at_k(scorer, valid_df, X_seen, allowed_items, users, k=10, batch_size=256, desc=None):
    truth = build_truth(valid_df)
    users = np.array([int(u) for u in users if int(u) in truth], dtype=np.int32)
    if len(users) == 0:
        return float('nan')
    allowed_items = np.asarray(allowed_items, dtype=np.int32)
    recalls = []
    iterator = range(0, len(users), batch_size)
    if desc:
        iterator = tqdm(iterator, desc=desc, leave=False)
    for st in iterator:
        batch_users = users[st:st + batch_size]
        scores = sanitize_scores(np.asarray(scorer.score_batch(batch_users), dtype=np.float32), bad_value=-1000000.0)
        scores_allowed = scores[:, allowed_items].copy()
        for r, u in enumerate(batch_users):
            seen = X_seen[int(u)].indices
            if len(seen):
                # Do not give credit for recommending items already present in the context.
                scores_allowed[r, np.isin(allowed_items, seen, assume_unique=False)] = -np.inf
            row = scores_allowed[r]
            finite = np.isfinite(row)
            if not np.any(finite):
                recalls.append(0.0)
                continue
            kk = min(k, int(finite.sum()))
            pos = np.where(finite)[0]
            vals = row[pos]
            if len(pos) > kk:
                part = np.argpartition(-vals, kk - 1)[:kk]
                top_items = allowed_items[pos[part][np.argsort(-vals[part])]]
            else:
                top_items = allowed_items[pos[np.argsort(-vals)]]
            t = truth.get(int(u), set())
            recalls.append(len(set((int(x) for x in top_items[:k])) & t) / min(k, len(t)))
    return float(np.mean(recalls)) if recalls else float('nan')

def evaluate_and_record(rows, name, group, scorer, data, X_seen, allowed_items, all_users, target_users, k, batch_size, notes=''):
    t0 = time.time()
    r_all = recall_at_k(scorer, data.valid, X_seen, allowed_items, all_users, k, batch_size, desc=name)
    r_target = np.nan
    if target_users is not None and len(target_users) > 0:
        r_target = recall_at_k(scorer, data.valid, X_seen, allowed_items, target_users, k, batch_size, desc=f'{name}:target')
    row = {'experiment': name, 'group': group, 'recall_at_10_all_valid_users': r_all, 'recall_at_10_target_overlap_users': r_target, 'time_sec': time.time() - t0, 'notes': notes}
    rows.append(row)
    print(f'{name:48s} | all={r_all:.6f}' + (f' | target={r_target:.6f}' if not np.isnan(r_target) else '') + f" | {row['time_sec']:.1f}s")
    return row

def parse_alpha_grid(s):
    return [float(x.strip()) for x in s.split(',') if x.strip()]

def score_health_check(name, scorer, users, n_check=64):
    users = np.array(list(users)[:n_check], dtype=np.int32)
    if len(users) == 0:
        return
    try:
        s = np.asarray(scorer.score_batch(users), dtype=np.float32)
        finite = np.isfinite(s).mean()
        print(f'Score health {name}: shape={s.shape}, finite={finite:.4f}, min={np.nanmin(s):.4g}, max={np.nanmax(s):.4g}, std={np.nanstd(s):.4g}')
    except Exception as e:
        print(f'Score health {name}: failed with {e!r}')

def evaluate_blend_grid(rows, base, addon, addon_name, data, X_seen, allowed_items, all_users, target_users, alpha_grid, k, batch_size, norm):
    best_alpha = None
    best = -1.0
    for alpha in alpha_grid:
        name = f'LightGCN + {addon_name} alpha={alpha:g}'
        scorer = BlendScorer(name, base, addon, alpha, norm)
        row = evaluate_and_record(rows, name, f'blend_{addon_name}', scorer, data, X_seen, allowed_items, all_users, target_users, k, batch_size, notes=f'{norm} score blend with alpha={alpha}')
        metric = row['recall_at_10_target_overlap_users']
        if np.isnan(metric):
            metric = row['recall_at_10_all_valid_users']
        if metric > best:
            best = float(metric)
            best_alpha = float(alpha)
    return (best_alpha, best)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--output_root', default='outputs')
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--validation', default='provided_test', choices=['provided_test', 'temporal'])
    ap.add_argument('--valid_fraction', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--reg', type=float, default=0.0001)
    ap.add_argument('--time_decay_half_life_days', type=int, default=365)
    ap.add_argument('--itemknn_topk', type=int, default=300)
    ap.add_argument('--seq_topk', type=int, default=300)
    ap.add_argument('--seq_recent_len', type=int, default=5)
    ap.add_argument('--include_sasrec', action='store_true', default=True)
    ap.add_argument('--skip_sasrec', action='store_true')
    ap.add_argument('--sasrec_epochs', type=int, default=80)
    ap.add_argument('--sasrec_hidden_dim', type=int, default=128)
    ap.add_argument('--sasrec_layers', type=int, default=2)
    ap.add_argument('--sasrec_heads', type=int, default=2)
    ap.add_argument('--sasrec_max_len', type=int, default=30)
    ap.add_argument('--sasrec_dropout', type=float, default=0.25)
    ap.add_argument('--sasrec_batch_size', type=int, default=512)
    ap.add_argument('--alpha_grid', default='0.02,0.05,0.1,0.2,0.35,0.5')
    ap.add_argument('--blend_norm', default='zscore', choices=['zscore', 'minmax', 'rrf', 'none'])
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--eval_batch_size', type=int, default=256)
    ap.add_argument('--max_eval_users', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.skip_sasrec:
        args.include_sasrec = False
    if args.quick:
        args.epochs = min(args.epochs, 10)
        args.sasrec_epochs = min(args.sasrec_epochs, 10)
        args.max_eval_users = args.max_eval_users or 1000
        args.itemknn_topk = min(args.itemknn_topk, 100)
        args.seq_topk = min(args.seq_topk, 100)
        print('Running in --quick mode.')
    seed_everything(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    out_dir = Path(args.output_root) / f'lightgcn_stepwise_{args.run_id or now_run_id()}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print('Output dir:', out_dir.resolve())
    print('Device:', device)
    data = load_encoded_data(Path(args.data_dir), args.validation, args.valid_fraction)
    X_binary = make_interaction_matrix(data.train_context, data.n_users, data.n_items, binary_after_sum=True)
    allowed_items = data.train_item_indices
    all_users = np.array(sorted(data.valid.user_idx.unique()), dtype=np.int32)
    if args.max_eval_users is not None and len(all_users) > args.max_eval_users:
        rng = np.random.default_rng(args.seed)
        all_users = np.array(sorted(rng.choice(all_users, size=args.max_eval_users, replace=False)), dtype=np.int32)
    target_users = None
    if data.sample_submission is not None and 'user_idx' in data.sample_submission.columns:
        target_users = np.array(sorted(set(all_users) & set(data.sample_submission.user_idx.astype(int))), dtype=np.int32)
    eda = {'validation_name': data.validation_name, 'n_users': data.n_users, 'n_items': data.n_items, 'train_all_rows': int(len(data.train_all)), 'train_context_rows': int(len(data.train_context)), 'valid_rows': int(len(data.valid)), 'valid_users': int(data.valid.user_idx.nunique()), 'eval_users': int(len(all_users)), 'target_overlap_eval_users': int(len(target_users)) if target_users is not None else 0, 'train_context_items': int(len(allowed_items)), 'recall_upper_bound_all': estimate_recall_upper_bound(data.valid, X_binary, allowed_items, all_users, args.k), 'recall_upper_bound_target': estimate_recall_upper_bound(data.valid, X_binary, allowed_items, target_users, args.k) if target_users is not None and len(target_users) > 0 else None}
    with open(out_dir / 'eda.json', 'w') as f:
        json.dump(eda, f, indent=2)
    print(json.dumps(eda, indent=2))
    results = []
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    best_alphas = {}
    scorers = {}
    # Start from a plain LightGCN, then add one signal at a time to see what actually helps.
    lightgcn_plain = train_lightgcn(LightGCNConfig('LightGCN_plain', args.dim, args.layers, args.epochs, args.batch_size, args.lr, args.reg, args.seed, 'uniform'), X_binary, X_binary, data.n_users, data.n_items, device)
    scorers['LightGCN_plain'] = lightgcn_plain
    evaluate_and_record(results, 'LightGCN_plain', 'baseline', lightgcn_plain, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'Plain binary user-item graph with uniform negatives.')
    simple = [('recent_pop_180d', VectorScorer('recent_pop_180d', time_weighted_item_scores(data.train_context, data.n_items, half_life_days=180)), 'Timestamp-only recent popularity.'), ('recent_pop_365d', VectorScorer('recent_pop_365d', time_weighted_item_scores(data.train_context, data.n_items, half_life_days=365)), 'Timestamp-only recent popularity.'), ('category_pop_365d', CategoryPopularityScorer('category_pop_365d', data.train_context, data.item_static, data.n_items, 365), 'Metadata category-aware recent popularity.')]
    for name, scorer, note in simple:
        scorers[name] = scorer
        evaluate_and_record(results, name, 'standalone_signal', scorer, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, note)
        best_alphas[name], _ = evaluate_blend_grid(results, lightgcn_plain, scorer, name, data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)
    W_itemknn = build_itemknn_cosine(X_binary, args.itemknn_topk)
    itemknn = SparseLinearScorer('itemknn_cosine', X_binary, W_itemknn)
    scorers['itemknn_cosine'] = itemknn
    evaluate_and_record(results, 'itemknn_cosine', 'standalone_signal', itemknn, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'Local item-item co-visitation signal.')
    best_alphas['itemknn_cosine'], _ = evaluate_blend_grid(results, lightgcn_plain, itemknn, 'itemknn_cosine', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)
    W_seq = build_sequence_transition_matrix(data.train_context, data.n_items, args.seq_topk)
    H_recent = build_recent_history_matrix(data.train_context, data.n_users, data.n_items, args.seq_recent_len)
    seq_transition = SparseLinearScorer('sequence_transition', H_recent, W_seq)
    scorers['sequence_transition'] = seq_transition
    evaluate_and_record(results, 'sequence_transition', 'standalone_signal', seq_transition, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'Directional next-item transition from timestamp-ordered histories.')
    best_alphas['sequence_transition'], _ = evaluate_blend_grid(results, lightgcn_plain, seq_transition, 'sequence_transition', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)
    content = ContentTfidfScorer('content_tfidf', X_binary, data.item_static, data.n_items, quick=args.quick)
    scorers['content_tfidf'] = content
    evaluate_and_record(results, 'content_tfidf', 'standalone_signal', content, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'TF-IDF using item_meta text fields only.')
    best_alphas['content_tfidf'], _ = evaluate_blend_grid(results, lightgcn_plain, content, 'content_tfidf', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)
    X_decay = make_interaction_matrix(data.train_context, data.n_users, data.n_items, weights=time_decay_weights(data.train_context, args.time_decay_half_life_days))
    lightgcn_time = train_lightgcn(LightGCNConfig(f'LightGCN_time_graph_hl{args.time_decay_half_life_days}', args.dim, args.layers, args.epochs, args.batch_size, args.lr, args.reg, args.seed + 11, 'uniform'), X_binary, X_decay, data.n_users, data.n_items, device)
    scorers['LightGCN_time_graph'] = lightgcn_time
    evaluate_and_record(results, lightgcn_time.name, 'lightgcn_variant', lightgcn_time, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'LightGCN with recency-weighted graph propagation.')
    best_alphas['LightGCN_time_graph'], _ = evaluate_blend_grid(results, lightgcn_plain, lightgcn_time, 'LightGCN_time_graph', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)
    lightgcn_popneg = train_lightgcn(LightGCNConfig('LightGCN_popular_negatives', args.dim, args.layers, args.epochs, args.batch_size, args.lr, args.reg, args.seed + 22, 'popularity'), X_binary, X_binary, data.n_users, data.n_items, device)
    scorers['LightGCN_popular_negatives'] = lightgcn_popneg
    evaluate_and_record(results, 'LightGCN_popular_negatives', 'lightgcn_variant', lightgcn_popneg, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'Same graph, but negatives sampled by item popularity^0.75.')
    best_alphas['LightGCN_popular_negatives'], _ = evaluate_blend_grid(results, lightgcn_plain, lightgcn_popneg, 'LightGCN_popular_negatives', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)
    if args.include_sasrec:
        sasrec = train_sasrec('SASRec', data.train_context, data.n_items, device, args.sasrec_epochs, args.sasrec_hidden_dim, args.sasrec_layers, args.sasrec_heads, args.sasrec_max_len, args.sasrec_dropout, args.sasrec_batch_size, args.seed + 33)
        scorers['SASRec'] = sasrec
        score_health_check('SASRec', sasrec, all_users)
        evaluate_and_record(results, 'SASRec', 'standalone_big_model', sasrec, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'Transformer sequential recommender trained from scratch.')
        best_alphas['SASRec'], _ = evaluate_blend_grid(results, lightgcn_plain, sasrec, 'SASRec', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)
    combo_specs = []
    if 'SASRec' in scorers:
        combo_specs.append(('LightGCN + SASRec + sequence', [(lightgcn_plain, 1.0), (scorers['SASRec'], best_alphas.get('SASRec', 0.1)), (seq_transition, best_alphas.get('sequence_transition', 0.1))]))
        combo_specs.append(('LightGCN + SASRec + time_graph', [(lightgcn_plain, 1.0), (scorers['SASRec'], best_alphas.get('SASRec', 0.1)), (lightgcn_time, best_alphas.get('LightGCN_time_graph', 0.1))]))
    combo_specs.append(('LightGCN + sequence + content', [(lightgcn_plain, 1.0), (seq_transition, best_alphas.get('sequence_transition', 0.1)), (content, best_alphas.get('content_tfidf', 0.1))]))
    combo_specs.append(('LightGCN + recent_pop + sequence', [(lightgcn_plain, 1.0), (simple[1][1], best_alphas.get('recent_pop_365d', 0.1)), (seq_transition, best_alphas.get('sequence_transition', 0.1))]))
    for name, comps in combo_specs:
        evaluate_and_record(results, name, 'controlled_combo', WeightedBlendScorer(name, comps, args.blend_norm), data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'Controlled multi-signal blend using best individual alpha values; not a reranker.')
    df = pd.DataFrame(results).sort_values(['recall_at_10_target_overlap_users', 'recall_at_10_all_valid_users'], ascending=[False, False], na_position='last')
    df.to_csv(out_dir / 'results.csv', index=False)
    with open(out_dir / 'best_alphas.json', 'w') as f:
        json.dump(best_alphas, f, indent=2)
    print('\n=== Best results ===')
    print(df.head(20).to_string(index=False))
    print('\nSaved:', out_dir / 'results.csv', out_dir / 'best_alphas.json', out_dir / 'eda.json', out_dir / 'config.json', sep='\n ')
if __name__ == '__main__':
    main()
