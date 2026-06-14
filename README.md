# Recommender Systems Assignment 4 Code

Code for the reported experiments and final Kaggle submission. The code uses only the provided assignment files: `train.csv`, `test.csv`, `item_meta.csv`, and `sample_submission.csv`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the provided CSV files in a `data/` directory:

```text
data/
  train.csv
  test.csv
  item_meta.csv
  sample_submission.csv
```

## Reproducing the reported experiments

Run commands from the project root.

### 1. Stand-alone baselines and controlled LightGCN additions

```bash
python 01_lightgcn_expansion_experiments.py \
  --data_dir data \
  --output_root outputs \
  --validation provided_test
```

### 2. Focused feature-addition experiments

```bash
python 02_feature_addition_experiments.py \
  --data_dir data \
  --output_root outputs \
  --validation provided_test \
  --dim 512 \
  --layers 4 \
  --epochs 220 \
  --seeds 42,43,44 \
  --recent_half_life 180
```

### 3. Final half-life and alpha validation sweep

```bash
python 03_finetune_validation_sweep.py \
  --data_dir data \
  --output_root outputs \
  --validation provided_test \
  --dim 512 \
  --layers 4 \
  --epochs 220 \
  --seeds 42,43,44 \
  --recent_half_lives 90,105,110,115,120,125,135,150,180,210,240 \
  --alphas 0.475,0.5,0.525
```

## Creating the final submission

```bash
python 04_train_submission.py \
  --data_dir data \
  --output_root outputs \
  --run_id final \
  --dim 512 \
  --layers 4 \
  --epochs 220 \
  --batch_size 4096 \
  --lr 0.001 \
  --reg 0.0002 \
  --seeds 42,43,44 \
  --recent_half_lives 120 \
  --alphas 0.5 \
  --blend_norm rrf \
  --prediction_batch_size 64
```

The submission CSV is written to `outputs/submission_lightgcn_recent_final/`.

## Files

- `00_data_analysis.ipynb`: data analysis.
- `01_lightgcn_expansion_experiments.py`: stand-alone baselines and controlled additions to LightGCN.
- `02_feature_addition_experiments.py`: multi-seed, trend, dynamic-alpha, recent-window, and category-popularity experiments.
- `03_finetune_validation_sweep.py`: final validation sweeps over half-life, alpha, and LightGCN capacity.
- `04_train_submission.py`: trains on all training interactions and writes Kaggle submission files.