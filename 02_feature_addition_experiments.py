#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from recommender_common.data import load_encoded_data
from recommender_common.lightgcn import Embeddings, TrainConfig, train_lightgcn
from recommender_common.matrices import make_interaction_matrix
from recommender_common.scoring import (
    CategoryConditionedPopularityScorer,
    DynamicAlphaBlendScorer,
    LightGCNScorer,
    MultiSeedLightGCNScorer,
    VectorScorer,
    WeightedBlendScorer,
    evaluate_and_record,
)
from recommender_common.signals import build_item_category_codes, time_weighted_item_scores, trend_score
from recommender_common.utils import (
    MS_PER_DAY,
    now_run_id,
    parse_dynamic_alpha_specs,
    parse_float_list,
    parse_int_list,
    parse_trend_specs,
    seed_everything,
    spec_to_name,
)


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
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
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
    cfg = TrainConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)

    full_embeddings: List[Embeddings] = []
    for seed in seeds:
        full_embeddings.append(train_lightgcn('LightGCN_full', seed=seed, cfg=cfg, X_binary=X_context, n_users=data.n_users, n_items=data.n_items, device=device))

    full_single = LightGCNScorer('LightGCN_full_seed' + str(seeds[0]), full_embeddings[0])
    full_multi = MultiSeedLightGCNScorer('LightGCN_full_multiseed_' + '-'.join(map(str, seeds)), full_embeddings, rrf_k=args.rrf_k)
    recent_pop = VectorScorer(f'recent_pop_{args.recent_half_life}d', time_weighted_item_scores(data.train_context, data.n_items, args.recent_half_life))
    hist_len = np.diff(X_context.indptr)
    evaluate_and_record(rows, full_single.name, 'baseline', full_single, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Single-seed full LightGCN.')
    evaluate_and_record(rows, recent_pop.name, 'standalone_signal', recent_pop, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Recent popularity standalone.')

    for alpha in alpha_grid:
        scorer = WeightedBlendScorer(f'single_LightGCN + recent{args.recent_half_life} alpha={alpha:g}', [(full_single, 1.0 - alpha), (recent_pop, alpha)], norm=args.blend_norm, rrf_k=args.rrf_k)
        evaluate_and_record(rows, scorer.name, 'current_family_single_seed', scorer, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Fixed-alpha current best family.')
    for spec in dynamic_specs:
        scorer = DynamicAlphaBlendScorer(f'dynamic_alpha_single_{spec_to_name(spec)}', full_single, recent_pop, hist_len=hist_len, alpha_bins=spec, norm=args.blend_norm, rrf_k=args.rrf_k)
        evaluate_and_record(rows, scorer.name, 'addition_dynamic_alpha', scorer, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'User-history-length dependent alpha.')

    if len(full_embeddings) > 1:
        evaluate_and_record(rows, full_multi.name, 'addition_multiseed', full_multi, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'RRF-average of multiple LightGCN seeds.')
        for alpha in alpha_grid:
            scorer = WeightedBlendScorer(f'multiseed_LightGCN + recent{args.recent_half_life} alpha={alpha:g}', [(full_multi, 1.0 - alpha), (recent_pop, alpha)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_multiseed', scorer, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed LightGCN plus recent popularity.')
        for spec in dynamic_specs:
            scorer = DynamicAlphaBlendScorer(f'dynamic_alpha_multiseed_{spec_to_name(spec)}', full_multi, recent_pop, hist_len=hist_len, alpha_bins=spec, norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_dynamic_alpha_multiseed', scorer, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Dynamic alpha using multi-seed LightGCN.')

    recent_cfg = TrainConfig(dim=args.dim, layers=args.layers, epochs=args.recent_lgcn_epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)
    for window in recent_windows:
        recent_df = recent_window_df(data.train_context, window)
        X_recent = make_interaction_matrix(recent_df, data.n_users, data.n_items, binary_after_sum=True)
        recent_embeddings = []
        for seed in recent_lgcn_seeds:
            recent_embeddings.append(train_lightgcn(f'LightGCN_recent{window}d', seed=seed, cfg=recent_cfg, X_binary=X_recent, n_users=data.n_users, n_items=data.n_items, device=device))
        if len(recent_embeddings) == 1:
            recent_lgcn = LightGCNScorer(f'LightGCN_recent{window}d_seed{recent_embeddings[0].seed}', recent_embeddings[0])
        else:
            recent_lgcn = MultiSeedLightGCNScorer(f'LightGCN_recent{window}d_multiseed_' + '-'.join(map(str, recent_lgcn_seeds)), recent_embeddings, rrf_k=args.rrf_k)
        evaluate_and_record(rows, recent_lgcn.name, 'addition_recent_window_lgcn', recent_lgcn, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, f'LightGCN trained only on last {window} days.')
        for gamma in extra_weight_grid:
            scorer = WeightedBlendScorer(f'single_LightGCN + recent_pop + recentLGCN{window} gamma={gamma:g}', [(full_single, 0.5), (recent_pop, 0.5), (recent_lgcn, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_recent_window_lgcn', scorer, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Base model plus recent-window LightGCN.')
            if len(full_embeddings) > 1:
                scorer2 = WeightedBlendScorer(f'multiseed_LightGCN + recent_pop + recentLGCN{window} gamma={gamma:g}', [(full_multi, 0.5), (recent_pop, 0.5), (recent_lgcn, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
                evaluate_and_record(rows, scorer2.name, 'addition_recent_window_lgcn_multiseed', scorer2, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed current best plus recent-window LightGCN.')
        del X_recent, recent_embeddings
        gc.collect()

    for short_hl, long_hl in trend_specs:
        trend = VectorScorer(f'trend_{short_hl}v{long_hl}', trend_score(data.train_context, data.n_items, short_hl=short_hl, long_hl=long_hl))
        evaluate_and_record(rows, trend.name, 'addition_trend_standalone', trend, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Relative trend / acceleration standalone.')
        for gamma in extra_weight_grid:
            scorer = WeightedBlendScorer(f'single_LightGCN + recent_pop + trend{short_hl}v{long_hl} gamma={gamma:g}', [(full_single, 0.5), (recent_pop, 0.5), (trend, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_trend', scorer, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Base model plus trend acceleration.')
            if len(full_embeddings) > 1:
                scorer2 = WeightedBlendScorer(f'multiseed_LightGCN + recent_pop + trend{short_hl}v{long_hl} gamma={gamma:g}', [(full_multi, 0.5), (recent_pop, 0.5), (trend, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
                evaluate_and_record(rows, scorer2.name, 'addition_trend_multiseed', scorer2, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed current best plus trend acceleration.')

    if data.item_meta is not None:
        item_cat = build_item_category_codes(data.item_meta, data.item2idx, data.n_items)
        cat_pop = CategoryConditionedPopularityScorer(f'category_conditioned_recent{args.recent_half_life}', data.train_context, item_cat, n_items=data.n_items, half_life_days=args.recent_half_life, global_weight=0.2)
        evaluate_and_record(rows, cat_pop.name, 'addition_category_conditioned_standalone', cat_pop, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'User dominant-category conditioned recent popularity.')
        for gamma in extra_weight_grid:
            scorer = WeightedBlendScorer(f'single_LightGCN + recent_pop + cat_recent gamma={gamma:g}', [(full_single, 0.5), (recent_pop, 0.5), (cat_pop, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
            evaluate_and_record(rows, scorer.name, 'addition_category_conditioned', scorer, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Base model plus user-category-conditioned recent popularity.')
            if len(full_embeddings) > 1:
                scorer2 = WeightedBlendScorer(f'multiseed_LightGCN + recent_pop + cat_recent gamma={gamma:g}', [(full_multi, 0.5), (recent_pop, 0.5), (cat_pop, gamma)], norm=args.blend_norm, rrf_k=args.rrf_k)
                evaluate_and_record(rows, scorer2.name, 'addition_category_conditioned_multiseed', scorer2, data, X_context, data.train_item_indices, all_valid_users, target_valid_users, args.k, args.eval_batch_size, 'Multi-seed current best plus category-conditioned recent popularity.')
    else:
        print('No item_meta.csv found; skipping category-conditioned recent popularity.')

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / 'results.csv', index=False)
    sorted_results = results.sort_values(['recall_at_10_all_valid_users', 'recall_at_10_target_overlap_users'], ascending=[False, False], na_position='last')
    sorted_results.to_csv(out_dir / 'results_sorted.csv', index=False)
    sorted_target = results.sort_values(['recall_at_10_target_overlap_users', 'recall_at_10_all_valid_users'], ascending=[False, False], na_position='last')
    sorted_target.to_csv(out_dir / 'results_sorted_target.csv', index=False)
    print('\n=== Top by all-valid Recall@10 ===')
    print(sorted_results.head(25).to_string(index=False))
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
