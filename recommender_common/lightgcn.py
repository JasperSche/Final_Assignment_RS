from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from tqdm.auto import tqdm

from .matrices import build_norm_adj
from .utils import seed_everything


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


@dataclass
class TrainConfig:
    dim: int
    layers: int
    epochs: int
    batch_size: int
    lr: float
    reg: float
    neg_sampling: str = 'uniform'
    time_decay_graph_half_life: Optional[int] = None


@dataclass
class Embeddings:
    seed: int
    user_emb: np.ndarray
    item_emb: np.ndarray
    name: str = 'LightGCN'


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


def build_popularity_sampler(X_binary: sparse.csr_matrix, power: float = 0.75) -> np.ndarray:
    counts = np.asarray(X_binary.sum(axis=0)).ravel().astype(np.float64)
    probs = np.power(counts + 1e-06, power)
    return probs / probs.sum()


def sample_bpr_batch(active_users: np.ndarray, pos_arrays: Dict[int, np.ndarray], pos_sets: Dict[int, set], n_items: int, batch_size: int, rng: np.random.Generator, neg_probs: Optional[np.ndarray] = None):
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
    return (users.astype(np.int64), pos, neg)


def train_lightgcn(name: str, seed: int, cfg: TrainConfig, X_binary: sparse.csr_matrix, n_users: int, n_items: int, device: torch.device, X_graph: Optional[sparse.csr_matrix] = None) -> Embeddings:
    seed_everything(seed)
    graph = X_binary if X_graph is None else X_graph
    print(f'\nTraining {name} seed={seed} dim={cfg.dim} layers={cfg.layers} epochs={cfg.epochs} neg={cfg.neg_sampling}')
    active_users, pos_arrays, pos_sets = prepare_user_positive_arrays(X_binary)
    if len(active_users) == 0:
        raise ValueError(f'{name} has no active users.')
    neg_probs = build_popularity_sampler(X_binary) if cfg.neg_sampling == 'popularity' else None
    norm_adj = build_norm_adj(graph, n_users, n_items, device)
    model = TorchLightGCN(n_users, n_items, cfg.dim, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, X_binary.nnz // cfg.batch_size)
    for epoch in tqdm(range(1, cfg.epochs + 1), desc=f'{name} seed={seed}'):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            users, pos, neg = sample_bpr_batch(active_users, pos_arrays, pos_sets, n_items, cfg.batch_size, rng, neg_probs=neg_probs)
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
    return Embeddings(seed=seed, user_emb=user_emb, item_emb=item_emb, name=name)

