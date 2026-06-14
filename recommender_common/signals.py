from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .utils import MS_PER_DAY


def time_decay_weights(df: pd.DataFrame, half_life_days: int) -> np.ndarray:
    max_ts = df['timestamp'].max()
    age_days = (max_ts - df['timestamp'].to_numpy()) / MS_PER_DAY
    return np.exp(-np.log(2) * age_days / half_life_days).astype(np.float32)


def time_weighted_counts(train_df: pd.DataFrame, n_items: int, half_life_days: Optional[int] = None, since_days: Optional[int] = None) -> np.ndarray:
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
    return np.bincount(df['item_idx'].to_numpy(np.int32), weights=weights, minlength=n_items).astype(np.float32)


def time_weighted_item_scores(train_df: pd.DataFrame, n_items: int, half_life_days: Optional[int] = None, since_days: Optional[int] = None) -> np.ndarray:
    return np.log1p(time_weighted_counts(train_df, n_items=n_items, half_life_days=half_life_days, since_days=since_days)).astype(np.float32)


def trend_score(train_df: pd.DataFrame, n_items: int, short_hl: int, long_hl: int) -> np.ndarray:
    short = time_weighted_counts(train_df, n_items=n_items, half_life_days=short_hl)
    long = time_weighted_counts(train_df, n_items=n_items, half_life_days=long_hl)
    return (np.log1p(short) - np.log1p(long + 1e-06) + 0.15 * np.log1p(short)).astype(np.float32)


def build_item_category_codes(item_meta: Optional[pd.DataFrame], item2idx: Dict[int, int], n_items: int) -> np.ndarray:
    if item_meta is None or 'item_id' not in item_meta.columns:
        return np.full(n_items, -1, dtype=np.int32)
    meta = item_meta.drop_duplicates('item_id').copy()
    if 'main_category' in meta.columns:
        cat_col = 'main_category'
    elif 'categories' in meta.columns:
        cat_col = 'categories'
    else:
        return np.full(n_items, -1, dtype=np.int32)
    meta = meta[['item_id', cat_col]].copy()
    meta['item_idx'] = meta['item_id'].map(item2idx)
    meta = meta.dropna(subset=['item_idx'])
    meta['item_idx'] = meta['item_idx'].astype('int32')
    meta[cat_col] = meta[cat_col].fillna('').astype(str)
    codes, _ = pd.factorize(meta[cat_col].replace('', np.nan), sort=True)
    meta['cat_code'] = codes.astype('int32')
    item_cat = np.full(n_items, -1, dtype=np.int32)
    valid = meta['cat_code'].to_numpy() >= 0
    item_cat[meta.loc[valid, 'item_idx'].to_numpy(np.int32)] = meta.loc[valid, 'cat_code'].to_numpy(np.int32)
    return item_cat

