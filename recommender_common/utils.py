from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

MS_PER_DAY = 1000 * 60 * 60 * 24
KEY_COLS = ['user_id', 'item_id', 'timestamp']


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def find_csv(data_dir: Path, canonical: str, required: bool = True) -> Optional[Path]:
    p = data_dir / canonical
    if p.exists():
        return p
    stem = canonical.replace('.csv', '')
    matches = sorted(data_dir.glob(f'{stem}*.csv'))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f'Could not find {canonical} or {stem}*.csv under {data_dir}')
    return None


def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == '':
        return []
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]


def parse_float_list(s: str) -> List[float]:
    if s is None or str(s).strip() == '':
        return []
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]


def parse_trend_specs(s: str) -> List[Tuple[int, int]]:
    if s is None or str(s).strip() == '':
        return []
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        short_hl, long_hl = part.split(':')
        out.append((int(short_hl), int(long_hl)))
    return out


def parse_dynamic_alpha_spec(spec: str) -> List[Tuple[int, float]]:
    bins = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        upper, alpha = part.split(':')
        bins.append((int(upper), float(alpha)))
    bins.sort(key=lambda x: x[0])
    if not bins:
        raise ValueError('Empty dynamic alpha spec.')
    return bins


def parse_dynamic_alpha_specs(s: str) -> List[List[Tuple[int, float]]]:
    specs = []
    for spec in str(s).split(';'):
        spec = spec.strip()
        if spec:
            specs.append(parse_dynamic_alpha_spec(spec))
    return specs


def alpha_for_history_len(hist_len: int, bins) -> float:
    for upper, alpha in bins:
        if hist_len <= upper:
            return alpha
    return bins[-1][1]


def spec_to_name(spec) -> str:
    return '_'.join([f"le{u}_a{safe_name(a)}" for u, a in spec])


def safe_name(x) -> str:
    return str(x).replace('.', 'p').replace('-', 'm').replace(',', '_')

