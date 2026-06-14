from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import KEY_COLS, find_csv


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


@dataclass
class SubmissionData:
    train: pd.DataFrame
    sample_submission: pd.DataFrame
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    idx2user: Dict[int, int]
    idx2item: Dict[int, int]
    n_users: int
    n_items: int
    train_item_indices: np.ndarray


def add_indices(df: pd.DataFrame, user2idx: Dict[int, int], item2idx: Dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out['user_idx'] = out['user_id'].map(user2idx).astype('int32')
    out['item_idx'] = out['item_id'].map(item2idx).astype('int32')
    return out


def split_provided_test(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    valid_keys = test[KEY_COLS].drop_duplicates().assign(_valid_row=1)
    overlap = test[KEY_COLS].drop_duplicates().merge(
        train[KEY_COLS].drop_duplicates().assign(_in_train=1),
        on=KEY_COLS,
        how='left',
    )['_in_train'].fillna(0).mean()
    merged = train.merge(valid_keys, on=KEY_COLS, how='left')
    train_context = merged[merged['_valid_row'].isna()].drop(columns=['_valid_row']).copy()
    valid = test.copy()
    warm_users = set(train_context.user_idx.unique())
    return (train_context, valid[valid.user_idx.isin(warm_users)].copy(), float(overlap))


def split_temporal_fraction(train: pd.DataFrame, valid_fraction: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = train['timestamp'].quantile(1.0 - valid_fraction)
    train_context = train[train['timestamp'] < cutoff].copy()
    valid = train[train['timestamp'] >= cutoff].copy()
    warm_users = set(train_context.user_idx.unique())
    return (train_context, valid[valid.user_idx.isin(warm_users)].copy())


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
            print(f'WARNING: provided test overlap only {overlap:.2%}; using temporal split.')
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
    return EncodedData(train, train_context, valid, sample, item_meta, user2idx, item2idx, idx2user, idx2item, len(all_user_ids), len(all_item_ids), train_item_indices, validation_name)


def load_submission_data(data_dir: Path) -> SubmissionData:
    train_path = find_csv(data_dir, 'train.csv', required=True)
    sample_path = find_csv(data_dir, 'sample_submission.csv', required=True)
    train_raw = pd.read_csv(train_path).drop_duplicates(KEY_COLS).copy()
    sample = pd.read_csv(sample_path).copy()
    train_raw['timestamp'] = pd.to_numeric(train_raw['timestamp'], errors='coerce').astype('int64')
    all_user_ids = set(train_raw.user_id.unique()) | set(sample.user_id.unique())
    all_item_ids = set(train_raw.item_id.unique())
    all_user_ids = np.array(sorted((int(u) for u in all_user_ids)))
    all_item_ids = np.array(sorted((int(i) for i in all_item_ids)))
    user2idx = {int(u): i for i, u in enumerate(all_user_ids)}
    item2idx = {int(it): i for i, it in enumerate(all_item_ids)}
    idx2user = {i: int(u) for u, i in user2idx.items()}
    idx2item = {i: int(it) for it, i in item2idx.items()}
    train = train_raw.copy()
    train['user_idx'] = train['user_id'].map(user2idx).astype('int32')
    train['item_idx'] = train['item_id'].map(item2idx).astype('int32')
    sample['user_idx'] = sample['user_id'].map(user2idx)
    if sample['user_idx'].isna().any():
        missing = int(sample['user_idx'].isna().sum())
        raise ValueError(f'{missing} sample_submission users are not in the encoded user universe.')
    sample['user_idx'] = sample['user_idx'].astype('int32')
    train_item_indices = np.array(sorted(train.item_idx.unique()), dtype=np.int32)
    return SubmissionData(train, sample, user2idx, item2idx, idx2user, idx2item, len(all_user_ids), len(all_item_ids), train_item_indices)

