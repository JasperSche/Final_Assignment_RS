#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from recommender_common.data import load_submission_data
from recommender_common.lightgcn import TrainConfig, train_lightgcn
from recommender_common.matrices import make_interaction_matrix
from recommender_common.scoring import normalize_dense
from recommender_common.signals import time_decay_weights, time_weighted_item_scores, trend_score
from recommender_common.submission import generate_predictions
from recommender_common.utils import now_run_id, parse_dynamic_alpha_spec, parse_float_list, parse_int_list, parse_trend_specs, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_root', type=str, default='outputs')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--reg', type=float, default=0.0001)
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated LightGCN seeds, e.g. 42,43,44.')
    parser.add_argument('--neg_sampling', type=str, default='uniform', choices=['uniform', 'popularity'])
    parser.add_argument('--time_decay_graph_half_life', type=int, default=0, help='0 means binary graph. Positive value uses recency-weighted graph propagation.')
    parser.add_argument('--recent_half_lives', type=str, default='180')
    parser.add_argument('--alphas', type=str, default='0.5,0.6')
    parser.add_argument('--trend_specs', type=str, default='', help='Optional trend specs, e.g. 120:365,90:365.')
    parser.add_argument('--trend_gammas', type=str, default='', help='Optional trend weights, e.g. 0.1,0.2,0.3.')
    parser.add_argument('--trend_base_alphas', type=str, default='', help='Optional alpha list for trend submissions; defaults to --alphas.')
    parser.add_argument('--blend_norm', type=str, default='rrf', choices=['rrf', 'zscore', 'minmax', 'none'])
    parser.add_argument('--rrf_k', type=float, default=60.0)
    parser.add_argument('--prediction_batch_size', type=int, default=128)
    parser.add_argument('--dynamic_alpha', action='store_true')
    parser.add_argument('--dynamic_alpha_spec', type=str, default='3:0.75,6:0.65,12:0.55,999999:0.45', help='History-length alpha bins for --dynamic_alpha.')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    seed_everything(42)
    run_id = args.run_id or now_run_id()
    out_dir = Path(args.output_root) / f'submission_lightgcn_recent_{run_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)

    data = load_submission_data(Path(args.data_dir))
    print('Output dir:', out_dir.resolve())
    print('Device:', device)
    print(f'Users={data.n_users:,}, items={data.n_items:,}, train rows={len(data.train):,}, submission users={len(data.sample_submission):,}')

    X_binary = make_interaction_matrix(data.train, data.n_users, data.n_items, binary_after_sum=True)
    if args.time_decay_graph_half_life and args.time_decay_graph_half_life > 0:
        graph_weights = time_decay_weights(data.train, args.time_decay_graph_half_life)
        X_graph = make_interaction_matrix(data.train, data.n_users, data.n_items, weights=graph_weights, binary_after_sum=False)
        graph_name = f'timegraph{args.time_decay_graph_half_life}'
    else:
        X_graph = X_binary
        graph_name = 'binarygraph'

    seeds = parse_int_list(args.seeds)
    alphas = parse_float_list(args.alphas)
    recent_half_lives = parse_int_list(args.recent_half_lives)
    trend_specs = parse_trend_specs(args.trend_specs)
    trend_gammas = parse_float_list(args.trend_gammas)
    trend_base_alphas = parse_float_list(args.trend_base_alphas)
    cfg = TrainConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg, neg_sampling=args.neg_sampling, time_decay_graph_half_life=args.time_decay_graph_half_life if args.time_decay_graph_half_life > 0 else None)

    with open(out_dir / 'config.json', 'w') as f:
        json.dump({**vars(args), 'train_config': asdict(cfg), 'seeds_parsed': seeds}, f, indent=2)

    embeddings = []
    t0 = time.time()
    for seed in seeds:
        emb = train_lightgcn('LightGCN', seed=seed, cfg=cfg, X_binary=X_binary, X_graph=X_graph, n_users=data.n_users, n_items=data.n_items, device=device)
        embeddings.append(emb)

    pop_score_vectors = {hl: time_weighted_item_scores(data.train, n_items=data.n_items, half_life_days=hl) for hl in recent_half_lives}
    trend_score_vectors = {spec: trend_score(data.train, n_items=data.n_items, short_hl=spec[0], long_hl=spec[1]) for spec in trend_specs}
    name_prefix = f"lgcn_d{args.dim}_l{args.layers}_e{args.epochs}_{graph_name}_{args.neg_sampling}_seeds{'-'.join(map(str, seeds))}_{args.blend_norm}"
    dynamic_bins = parse_dynamic_alpha_spec(args.dynamic_alpha_spec) if args.dynamic_alpha else None
    output_paths = generate_predictions(data=data, X_seen=X_binary, embeddings=embeddings, pop_score_vectors=pop_score_vectors, alphas=alphas, blend_norm=args.blend_norm, rrf_k=args.rrf_k, batch_size=args.prediction_batch_size, output_dir=out_dir, name_prefix=name_prefix, normalize_dense_fn=normalize_dense, dynamic_alpha_bins=dynamic_bins, trend_score_vectors=trend_score_vectors, trend_gammas=trend_gammas, trend_base_alphas=trend_base_alphas)

    summary = {'run_id': run_id, 'output_dir': str(out_dir), 'num_outputs': len(output_paths), 'outputs': [str(p) for p in output_paths], 'train_time_sec': time.time() - t0}
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nDone. Outputs:')
    for p in output_paths:
        print(' ', p)
    print('Summary:', out_dir / 'summary.json')


if __name__ == '__main__':
    main()
