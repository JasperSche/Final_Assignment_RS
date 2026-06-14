#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from recommender_common.data import load_encoded_data
from recommender_common.lightgcn import Embeddings, TrainConfig, train_lightgcn
from recommender_common.matrices import make_interaction_matrix
from recommender_common.scoring import LightGCNScorer, VectorScorer, WeightedBlendScorer, recall_at_k
from recommender_common.signals import time_weighted_item_scores
from recommender_common.utils import now_run_id, parse_float_list, parse_int_list, seed_everything


def evaluate_blend(scorer, data, X_seen, users, target_users, k: int, batch_size: int) -> tuple[float, float]:
    recall_all = recall_at_k(scorer, data.valid, X_seen, data.train_item_indices, users, k=k, batch_size=batch_size, desc=scorer.name)
    recall_target = np.nan
    if target_users is not None and len(target_users):
        recall_target = recall_at_k(scorer, data.valid, X_seen, data.train_item_indices, target_users, k=k, batch_size=batch_size, desc=scorer.name + ':target')
    return (recall_all, recall_target)


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
    parser.add_argument('--recent_half_lives', type=str, default='120,150,180,210,240')
    parser.add_argument('--alphas', type=str, default='0.45,0.475,0.5,0.525,0.55')
    parser.add_argument('--blend_norm', type=str, default='rrf', choices=['rrf', 'zscore', 'minmax', 'none'])
    parser.add_argument('--rrf_k', type=float, default=60.0)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--max_eval_users', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        print('Running in --quick mode.')
        args.dim = min(args.dim, 128)
        args.epochs = min(args.epochs, 30)
        args.seeds = args.seeds.split(',')[0]
        args.max_eval_users = args.max_eval_users or 1000

    seed_everything(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    run_id = args.run_id or now_run_id()
    out_dir = Path(args.output_root) / f'lightgcn_finetune_validation_{run_id}'
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

    seeds = parse_int_list(args.seeds)
    half_lives = parse_int_list(args.recent_half_lives)
    alphas = parse_float_list(args.alphas)
    cfg = TrainConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)

    embeddings: List[Embeddings] = []
    for seed in seeds:
        embeddings.append(train_lightgcn('LightGCN', seed=seed, cfg=cfg, X_binary=X_context, n_users=data.n_users, n_items=data.n_items, device=device))

    rows = []
    pop_components = {hl: time_weighted_item_scores(data.train_context, data.n_items, hl) for hl in half_lives}
    lgcn_components = [(LightGCNScorer(f'LightGCN_seed{emb.seed}', emb), 1.0 / len(embeddings)) for emb in embeddings]
    for hl in half_lives:
        pop_scorer = VectorScorer(f'recent_pop_{hl}', pop_components[hl])
        for alpha in alphas:
            t0 = time.time()
            name = f"lgcn_d{args.dim}_l{args.layers}_e{args.epochs}_seeds{'-'.join(map(str, seeds))}_hl{hl}_alpha{alpha:g}"
            components = [(scorer, (1.0 - alpha) * weight) for scorer, weight in lgcn_components] + [(pop_scorer, alpha)]
            scorer = WeightedBlendScorer(name, components, norm=args.blend_norm, rrf_k=args.rrf_k)
            recall_all, recall_target = evaluate_blend(scorer, data, X_context, all_valid_users, target_valid_users, args.k, args.eval_batch_size)
            row = {'experiment': name, 'dim': args.dim, 'layers': args.layers, 'epochs': args.epochs, 'seeds': ','.join(map(str, seeds)), 'recent_half_life': hl, 'alpha': alpha, 'recall_at_10_all_valid_users': recall_all, 'recall_at_10_target_overlap_users': recall_target, 'time_sec': time.time() - t0}
            rows.append(row)
            print(f"{name:80s} | all={recall_all:.6f} | target={recall_target:.6f} | {row['time_sec']:.1f}s")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / 'results.csv', index=False)
    sorted_all = results.sort_values(['recall_at_10_all_valid_users', 'recall_at_10_target_overlap_users'], ascending=[False, False], na_position='last')
    sorted_target = results.sort_values(['recall_at_10_target_overlap_users', 'recall_at_10_all_valid_users'], ascending=[False, False], na_position='last')
    sorted_all.to_csv(out_dir / 'results_sorted.csv', index=False)
    sorted_target.to_csv(out_dir / 'results_sorted_target.csv', index=False)
    print('\n=== Top by all-valid Recall@10 ===')
    print(sorted_all.head(25).to_string(index=False))
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
