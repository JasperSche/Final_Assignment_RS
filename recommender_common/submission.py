from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm

from .data import SubmissionData
from .lightgcn import Embeddings
from .scoring import normalize_vector
from .utils import alpha_for_history_len, safe_name, spec_to_name


def make_lightgcn_component(embeddings: Sequence[Embeddings], user_indices: np.ndarray, blend_norm: str, rrf_k: float, normalize_dense_fn) -> np.ndarray:
    out = None
    for emb in embeddings:
        raw = emb.user_emb[user_indices] @ emb.item_emb.T
        comp = normalize_dense_fn(raw, method=blend_norm, rrf_k=rrf_k)
        out = comp if out is None else out + comp
    out /= float(len(embeddings))
    return out.astype(np.float32)


def build_submission_dataframe(sample: pd.DataFrame, pred_item_indices: Dict[int, List[int]], idx2item: Dict[int, int]) -> pd.DataFrame:
    rows = []
    has_id = 'ID' in sample.columns
    if 'prediction' in sample.columns:
        pred_col = 'prediction'
        sep = ' '
    else:
        pred_col = 'item_id'
        sep = ','
    for _, row in sample.iterrows():
        uidx = int(row['user_idx'])
        item_ids = [str(idx2item[int(i)]) for i in pred_item_indices[uidx]]
        out = {}
        if has_id:
            out['ID'] = row['ID']
        out['user_id'] = int(row['user_id'])
        out[pred_col] = sep.join(item_ids)
        rows.append(out)
    return pd.DataFrame(rows)


def generate_predictions(data: SubmissionData, X_seen: sparse.csr_matrix, embeddings: Sequence[Embeddings], pop_score_vectors: Dict[int, np.ndarray], alphas: Sequence[float], blend_norm: str, rrf_k: float, batch_size: int, output_dir: Path, name_prefix: str, normalize_dense_fn, dynamic_alpha_bins: Optional[List[Tuple[int, float]]] = None, trend_score_vectors: Optional[Dict[Tuple[int, int], np.ndarray]] = None, trend_gammas: Optional[Sequence[float]] = None, trend_base_alphas: Optional[Sequence[float]] = None) -> List[Path]:
    sample = data.sample_submission
    target_users = sample['user_idx'].to_numpy(np.int32)
    allowed_items = data.train_item_indices
    allowed_mask = np.zeros(data.n_items, dtype=bool)
    allowed_mask[allowed_items] = True
    hist_len = np.diff(X_seen.indptr)
    output_paths = []
    pop_components = {hl: normalize_vector(scores, method=blend_norm, rrf_k=rrf_k) for hl, scores in pop_score_vectors.items()}
    trend_components = {}
    if trend_score_vectors:
        trend_components = {spec: normalize_vector(scores, method=blend_norm, rrf_k=rrf_k) for spec, scores in trend_score_vectors.items()}
    for half_life, pop_component in pop_components.items():
        for alpha in alphas:
            print(f'\nGenerating submission: half_life={half_life}, alpha={alpha}')
            pred_item_indices: Dict[int, List[int]] = {}
            for start in tqdm(range(0, len(target_users), batch_size), desc=f'predict hl={half_life} a={alpha}', leave=False):
                batch_users = target_users[start:start + batch_size]
                lgcn_component = make_lightgcn_component(embeddings, batch_users, blend_norm, rrf_k, normalize_dense_fn)
                scores = (1.0 - alpha) * lgcn_component + alpha * pop_component[None, :]
                scores[:, ~allowed_mask] = -np.inf
                _fill_topk(pred_item_indices, scores, batch_users, X_seen)
            sub = build_submission_dataframe(sample, pred_item_indices, data.idx2item)
            out_path = output_dir / f'{name_prefix}_hl{half_life}_alpha{safe_name(alpha)}.csv'
            sub.to_csv(out_path, index=False)
            print('Saved', out_path)
            output_paths.append(out_path)
    if dynamic_alpha_bins is not None:
        for half_life, pop_component in pop_components.items():
            print(f'\nGenerating dynamic-alpha submission: half_life={half_life}, bins={dynamic_alpha_bins}')
            pred_item_indices = {}
            for start in tqdm(range(0, len(target_users), batch_size), desc=f'predict dynamic hl={half_life}', leave=False):
                batch_users = target_users[start:start + batch_size]
                lgcn_component = make_lightgcn_component(embeddings, batch_users, blend_norm, rrf_k, normalize_dense_fn)
                user_alphas = np.array([alpha_for_history_len(int(hist_len[int(u)]), dynamic_alpha_bins) for u in batch_users], dtype=np.float32)
                scores = (1.0 - user_alphas[:, None]) * lgcn_component + user_alphas[:, None] * pop_component[None, :]
                scores[:, ~allowed_mask] = -np.inf
                _fill_topk(pred_item_indices, scores, batch_users, X_seen)
            sub = build_submission_dataframe(sample, pred_item_indices, data.idx2item)
            out_path = output_dir / f'{name_prefix}_hl{half_life}_dynamic_{spec_to_name(dynamic_alpha_bins)}.csv'
            sub.to_csv(out_path, index=False)
            print('Saved', out_path)
            output_paths.append(out_path)
    if trend_components and trend_gammas:
        base_alphas = list(trend_base_alphas) if trend_base_alphas else list(alphas)
        for half_life, pop_component in pop_components.items():
            for alpha in base_alphas:
                for (short_hl, long_hl), trend_component in trend_components.items():
                    for gamma in trend_gammas:
                        print(f'\nGenerating trend submission: hl={half_life}, alpha={alpha}, trend={short_hl}v{long_hl}, gamma={gamma}')
                        pred_item_indices = {}
                        for start in tqdm(range(0, len(target_users), batch_size), desc=f'predict trend {short_hl}v{long_hl} g={gamma}', leave=False):
                            batch_users = target_users[start:start + batch_size]
                            lgcn_component = make_lightgcn_component(embeddings, batch_users, blend_norm, rrf_k, normalize_dense_fn)
                            scores = (1.0 - alpha) * lgcn_component + alpha * pop_component[None, :] + float(gamma) * trend_component[None, :]
                            scores[:, ~allowed_mask] = -np.inf
                            _fill_topk(pred_item_indices, scores, batch_users, X_seen)
                        sub = build_submission_dataframe(sample, pred_item_indices, data.idx2item)
                        out_path = output_dir / f'{name_prefix}_hl{half_life}_alpha{safe_name(alpha)}_trend{short_hl}v{long_hl}_gamma{safe_name(gamma)}.csv'
                        sub.to_csv(out_path, index=False)
                        print('Saved', out_path)
                        output_paths.append(out_path)
    return output_paths


def _fill_topk(pred_item_indices: Dict[int, List[int]], scores: np.ndarray, batch_users: np.ndarray, X_seen: sparse.csr_matrix, k: int = 10) -> None:
    for r, u in enumerate(batch_users):
        seen = X_seen[int(u)].indices
        if len(seen):
            scores[r, seen] = -np.inf
        row = scores[r]
        finite = np.isfinite(row)
        if finite.sum() < k:
            raise RuntimeError(f'User {u} has fewer than {k} finite candidate scores.')
        candidate_idx = np.where(finite)[0]
        candidate_scores = row[candidate_idx]
        part = np.argpartition(-candidate_scores, k - 1)[:k]
        top_items = candidate_idx[part[np.argsort(-candidate_scores[part])]]
        pred_item_indices[int(u)] = [int(x) for x in top_items]

