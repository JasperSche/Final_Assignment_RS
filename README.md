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

### Phase 1 and Phase 2: baselines and controlled additions

This reproduces the local stand-alone model comparison and the controlled additions to the initial LightGCN baseline.

```bash
python 01_lightgcn_expansion_experiments.py \
  --data_dir data \
  --output_root outputs \
  --validation provided_test \
  --valid_fraction 0.15 \
  --seed 42 \
  --dim 128 \
  --layers 3 \
  --epochs 160 \
  --batch_size 4096 \
  --lr 0.002 \
  --reg 0.0001 \
  --time_decay_half_life_days 365 \
  --itemknn_topk 500 \
  --seq_topk 500 \
  --seq_recent_len 5 \
  --sasrec_epochs 120 \
  --sasrec_hidden_dim 128 \
  --sasrec_layers 2 \
  --sasrec_heads 2 \
  --sasrec_max_len 30 \
  --sasrec_dropout 0.25 \
  --sasrec_batch_size 512 \
  --alpha_grid 0.002,0.005,0.01,0.02,0.035,0.05,0.075,0.1,0.15,0.2,0.3 \
  --blend_norm rrf \
  --k 10 \
  --eval_batch_size 128
```

### Phase 3: focused refinements

Local feature-addition experiments around the 512d/4-layer, three-seed LightGCN + recent-popularity model:

```bash
python 02_feature_addition_experiments.py \
  --data_dir data \
  --output_root outputs \
  --validation provided_test \
  --valid_fraction 0.15 \
  --seed 42 \
  --dim 512 \
  --layers 4 \
  --epochs 220 \
  --batch_size 4096 \
  --lr 0.001 \
  --reg 0.0002 \
  --seeds 42,43,44 \
  --recent_half_life 180 \
  --alpha_grid 0.42,0.45,0.475,0.5,0.525,0.55,0.575,0.6 \
  --blend_norm rrf \
  --rrf_k 60 \
  --dynamic_alpha_specs '3:0.65,6:0.55,12:0.50,999999:0.45;3:0.75,6:0.65,12:0.55,999999:0.45;3:0.60,6:0.55,12:0.50,999999:0.40' \
  --recent_lgcn_windows 180,365 \
  --recent_lgcn_epochs 160 \
  --recent_lgcn_seeds 42 \
  --trend_specs 30:365,60:365,90:365,120:365 \
  --extra_weight_grid 0.05,0.1,0.15,0.2,0.3 \
  --eval_batch_size 64 \
  --k 10
```

### Phase 4: final half-life and alpha tuning

Validation sweep for the final model family:

```bash
python 03_finetune_validation_sweep.py \
  --data_dir data \
  --output_root outputs \
  --validation provided_test \
  --valid_fraction 0.15 \
  --seed 42 \
  --dim 512 \
  --layers 4 \
  --epochs 220 \
  --batch_size 4096 \
  --lr 0.001 \
  --reg 0.0002 \
  --seeds 42,43,44 \
  --recent_half_lives 120,150,180,210,240 \
  --alphas 0.45,0.475,0.5,0.525,0.55 \
  --blend_norm rrf \
  --rrf_k 60 \
  --eval_batch_size 64 \
  --k 10
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