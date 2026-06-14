#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommender_common.data import load_encoded_data
from recommender_common.lightgcn import TrainConfig, train_lightgcn
from recommender_common.matrices import make_interaction_matrix, topk_sparse_rows
from recommender_common.scoring import (
    BlendScorer,
    LightGCNScorer,
    SparseLinearScorer,
    VectorScorer,
    WeightedBlendScorer,
    evaluate_and_record,
    recall_at_k,
    sanitize_scores,
)
from recommender_common.signals import time_decay_weights, time_weighted_item_scores
from recommender_common.utils import MS_PER_DAY, now_run_id, seed_everything


def coerce_text(x) -> str:
    return '' if pd.isna(x) else str(x)


def build_item_static(item_meta: pd.DataFrame, all_item_ids, item2idx: Dict[int, int]) -> pd.DataFrame:
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


class CategoryPopularityScorer:
    def __init__(self, name: str, train_df: pd.DataFrame, item_static: pd.DataFrame, n_items: int, half_life_days: int = 365):
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
    def __init__(self, name: str, X_user_item: sparse.csr_matrix, item_static: pd.DataFrame, n_items: int, quick: bool = False):
        self.name = name
        self.X = X_user_item.tocsr()
        self.n_items = n_items
        text = item_static.sort_values('item_idx')['text_for_content'].fillna('').astype(str).values
        max_word = 80000 if quick else 200000
        max_char = 40000 if quick else 100000
        self.vectorizer = FeatureUnion([('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True, dtype=np.float32)), ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=max_char, sublinear_tf=True, dtype=np.float32))])
        print('Fitting metadata TF-IDF content scorer...')
        self.F_items = normalize(self.vectorizer.fit_transform(text), norm='l2', axis=1).tocsr().astype(np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        P = normalize(self.X[user_indices] @ self.F_items, norm='l2', axis=1)
        S = P @ self.F_items.T
        return S.toarray().astype(np.float32) if sparse.issparse(S) else S.astype(np.float32)


def build_itemknn_cosine(X: sparse.csr_matrix, topk: int = 300) -> sparse.csr_matrix:
    print(f'Building itemKNN cosine topk={topk}')
    Y = normalize(X.T, norm='l2', axis=1)
    W = Y @ Y.T
    W = W.tolil()
    W.setdiag(0)
    W = W.tocsr()
    W.eliminate_zeros()
    return topk_sparse_rows(W, topk)


def build_sequence_transition_matrix(train_df: pd.DataFrame, n_items: int, topk: int = 300, max_gap_days: Optional[int] = None) -> sparse.csr_matrix:
    print(f'Building sequence transition matrix topk={topk}')
    rows = []
    cols = []
    data = []
    for _, g in tqdm(train_df.sort_values(['user_idx', 'timestamp']).groupby('user_idx'), desc='user sequences', leave=False):
        items = g['item_idx'].to_numpy(np.int32)
        ts = g['timestamp'].to_numpy(np.int64)
        if len(items) < 2:
            continue
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


def build_recent_history_matrix(train_df: pd.DataFrame, n_users: int, n_items: int, max_len: int = 5) -> sparse.csr_matrix:
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
    def __init__(self, train_df: pd.DataFrame, max_len: int, n_items: int, seed: int = 42, max_examples_per_user: Optional[int] = None):
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
    def __init__(self, n_items: int, max_len: int = 30, hidden_dim: int = 128, n_layers: int = 2, n_heads: int = 2, dropout: float = 0.25):
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


def estimate_recall_upper_bound(valid_df, X_seen, allowed_items, users, k):
    allowed = set((int(x) for x in allowed_items))
    truth = valid_df.groupby('user_idx')['item_idx'].apply(lambda x: set((int(v) for v in x.values))).to_dict()
    vals = []
    for u in users:
        t = truth.get(int(u), set())
        if not t:
            continue
        seen = set((int(x) for x in X_seen[int(u)].indices))
        reachable = [it for it in t if it in allowed and it not in seen]
        vals.append(min(len(reachable), k) / min(len(t), k))
    return float(np.mean(vals)) if vals else float('nan')


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
    all_item_ids = [data.idx2item[i] for i in range(data.n_items)]
    item_static = build_item_static(data.item_meta, all_item_ids, data.item2idx)
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
    cfg = TrainConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)
    lightgcn_emb = train_lightgcn('LightGCN_plain', seed=args.seed, cfg=cfg, X_binary=X_binary, X_graph=X_binary, n_users=data.n_users, n_items=data.n_items, device=device)
    lightgcn_plain = LightGCNScorer('LightGCN_plain', lightgcn_emb)
    scorers['LightGCN_plain'] = lightgcn_plain
    evaluate_and_record(results, 'LightGCN_plain', 'baseline', lightgcn_plain, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'Plain binary user-item graph with uniform negatives.')

    simple = [
        ('recent_pop_180d', VectorScorer('recent_pop_180d', time_weighted_item_scores(data.train_context, data.n_items, half_life_days=180)), 'Timestamp-only recent popularity.'),
        ('recent_pop_365d', VectorScorer('recent_pop_365d', time_weighted_item_scores(data.train_context, data.n_items, half_life_days=365)), 'Timestamp-only recent popularity.'),
        ('category_pop_365d', CategoryPopularityScorer('category_pop_365d', data.train_context, item_static, data.n_items, 365), 'Metadata category-aware recent popularity.'),
    ]
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

    content = ContentTfidfScorer('content_tfidf', X_binary, item_static, data.n_items, quick=args.quick)
    scorers['content_tfidf'] = content
    evaluate_and_record(results, 'content_tfidf', 'standalone_signal', content, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'TF-IDF using item_meta text fields only.')
    best_alphas['content_tfidf'], _ = evaluate_blend_grid(results, lightgcn_plain, content, 'content_tfidf', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)

    X_decay = make_interaction_matrix(data.train_context, data.n_users, data.n_items, weights=time_decay_weights(data.train_context, args.time_decay_half_life_days), binary_after_sum=False)
    time_cfg = TrainConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)
    time_emb = train_lightgcn(f'LightGCN_time_graph_hl{args.time_decay_half_life_days}', seed=args.seed + 11, cfg=time_cfg, X_binary=X_binary, X_graph=X_decay, n_users=data.n_users, n_items=data.n_items, device=device)
    lightgcn_time = LightGCNScorer(f'LightGCN_time_graph_hl{args.time_decay_half_life_days}', time_emb)
    scorers['LightGCN_time_graph'] = lightgcn_time
    evaluate_and_record(results, lightgcn_time.name, 'lightgcn_variant', lightgcn_time, data, X_binary, allowed_items, all_users, target_users, args.k, args.eval_batch_size, 'LightGCN with recency-weighted graph propagation.')
    best_alphas['LightGCN_time_graph'], _ = evaluate_blend_grid(results, lightgcn_plain, lightgcn_time, 'LightGCN_time_graph', data, X_binary, allowed_items, all_users, target_users, alpha_grid, args.k, args.eval_batch_size, args.blend_norm)

    popneg_cfg = TrainConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg, neg_sampling='popularity')
    popneg_emb = train_lightgcn('LightGCN_popular_negatives', seed=args.seed + 22, cfg=popneg_cfg, X_binary=X_binary, X_graph=X_binary, n_users=data.n_users, n_items=data.n_items, device=device)
    lightgcn_popneg = LightGCNScorer('LightGCN_popular_negatives', popneg_emb)
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
