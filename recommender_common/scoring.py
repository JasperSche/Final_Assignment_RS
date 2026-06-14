from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm

from .lightgcn import Embeddings
from .signals import time_weighted_item_scores
from .utils import alpha_for_history_len


def dense_rrf(scores: np.ndarray, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e+30, posinf=1e+30, neginf=-1e+30)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(scores.shape[1], dtype=np.int32)[None, :]
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)


def vector_rrf(scores: np.ndarray, rrf_k: float = 60.0) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e+30, posinf=1e+30, neginf=-1e+30)
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(len(scores), dtype=np.int32)
    return (1.0 / (rrf_k + ranks + 1.0)).astype(np.float32)


def normalize_dense(scores: np.ndarray, method: str = 'rrf', rrf_k: float = 60.0) -> np.ndarray:
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


def normalize_vector(scores: np.ndarray, method: str = 'rrf', rrf_k: float = 60.0) -> np.ndarray:
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


def sanitize_scores(scores: np.ndarray, bad_value: float = 0.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    return np.nan_to_num(scores, nan=bad_value, posinf=1000000.0, neginf=-1000000.0).astype(np.float32, copy=False)


def normalize_score_matrix(scores: np.ndarray, method: str = 'zscore') -> np.ndarray:
    scores = sanitize_scores(scores, bad_value=0.0)
    return normalize_dense(scores, method=method, rrf_k=60.0)


class LightGCNScorer:
    def __init__(self, name: str, emb: Embeddings):
        self.name = name
        self.emb = emb

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.emb.user_emb[user_indices] @ self.emb.item_emb.T).astype(np.float32)


class MultiSeedLightGCNScorer:
    def __init__(self, name: str, embeddings: Sequence[Embeddings], rrf_k: float = 60.0):
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


class SparseLinearScorer:
    def __init__(self, name: str, X_user_item: sparse.csr_matrix, W_item_item: sparse.csr_matrix):
        self.name = name
        self.X = X_user_item.tocsr()
        self.W = W_item_item.tocsr()

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        S = self.X[user_indices] @ self.W
        return S.toarray().astype(np.float32) if sparse.issparse(S) else S.astype(np.float32)


class WeightedBlendScorer:
    def __init__(self, name: str, components: Sequence[Tuple[object, float]], norm: str = 'rrf', rrf_k: float = 60.0):
        self.name = name
        self.components = list(components)
        self.norm = norm
        self.rrf_k = rrf_k

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        out = None
        for scorer, weight in self.components:
            s = normalize_dense(scorer.score_batch(user_indices), method=self.norm, rrf_k=self.rrf_k)
            out = weight * s if out is None else out + weight * s
        return out.astype(np.float32)


class BlendScorer:
    def __init__(self, name: str, base, addon, alpha: float, norm: str = 'zscore'):
        self.name = name
        self.base = base
        self.addon = addon
        self.alpha = float(alpha)
        self.norm = norm

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        a = normalize_score_matrix(self.base.score_batch(user_indices), self.norm)
        b = normalize_score_matrix(self.addon.score_batch(user_indices), self.norm)
        return ((1.0 - self.alpha) * a + self.alpha * b).astype(np.float32)


class DynamicAlphaBlendScorer:
    def __init__(self, name: str, base_scorer, pop_scorer, hist_len: np.ndarray, alpha_bins, norm: str = 'rrf', rrf_k: float = 60.0):
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
    def __init__(self, name: str, train_df: pd.DataFrame, item_cat: np.ndarray, n_items: int, half_life_days: int = 180, global_weight: float = 0.2):
        self.name = name
        self.n_items = n_items
        self.global_weight = float(global_weight)
        self.item_cat = item_cat
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


def build_truth(valid_df: pd.DataFrame) -> Dict[int, set]:
    return valid_df.groupby('user_idx')['item_idx'].apply(lambda x: set((int(v) for v in x.values))).to_dict()


def recall_at_k(scorer, valid_df: pd.DataFrame, X_seen: sparse.csr_matrix, allowed_items: np.ndarray, users: Sequence[int], k: int = 10, batch_size: int = 64, desc: Optional[str] = None) -> float:
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
            kk = min(k, int(finite.sum()))
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


def evaluate_and_record(rows: List[dict], name: str, group: str, scorer, data, X_seen: sparse.csr_matrix, allowed_items: np.ndarray, all_users: np.ndarray, target_users: Optional[np.ndarray], k: int, batch_size: int, notes: str = '') -> dict:
    t0 = time.time()
    r_all = recall_at_k(scorer, data.valid, X_seen, allowed_items, all_users, k=k, batch_size=batch_size, desc=name)
    r_target = np.nan
    if target_users is not None and len(target_users) > 0:
        r_target = recall_at_k(scorer, data.valid, X_seen, allowed_items, target_users, k=k, batch_size=batch_size, desc=f'{name}:target')
    row = {'experiment': name, 'group': group, 'recall_at_10_all_valid_users': r_all, 'recall_at_10_target_overlap_users': r_target, 'time_sec': time.time() - t0, 'notes': notes}
    rows.append(row)
    print(f'{name:48s} | all={r_all:.6f}' + (f' | target={r_target:.6f}' if not np.isnan(r_target) else '') + f" | {row['time_sec']:.1f}s")
    return row

