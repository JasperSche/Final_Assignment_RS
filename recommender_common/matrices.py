from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm.auto import tqdm


def make_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int, weights: Optional[np.ndarray] = None, binary_after_sum: bool = True) -> sparse.csr_matrix:
    rows = df['user_idx'].to_numpy(np.int32)
    cols = df['item_idx'].to_numpy(np.int32)
    vals = np.ones(len(df), dtype=np.float32) if weights is None else weights.astype(np.float32)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    X.sum_duplicates()
    if binary_after_sum:
        X.data[:] = 1.0
    return X


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

