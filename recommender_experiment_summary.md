# Recommender System Experiment Summary

## Goal

We started from a strong **plain LightGCN** baseline and tested whether additional signals or larger architectures could improve Recall@10 and leaderboard performance.

Final best submission:

```text
LightGCN 512d / 4 layers / 220 epochs
+ recent popularity, 180-day half-life
+ RRF blend, alpha = 0.5
Leaderboard score: 0.0236
```

This ranked #1 at the time of submission.

---

## Experiments Run

### 1. Complex Ensemble / Reranker

Tested a large multi-generator reranking setup with:

- LightGCN variants
- EASE
- RP3β
- ItemKNN
- Sequential models
- Metadata/content models
- Popularity features
- LightGBM-style reranking

**Result:** Too complex and memory-heavy. It did not clearly outperform plain LightGCN.

**Lesson:** More components made the system harder to debug without giving reliable gains.

---

### 2. Plain LightGCN Baseline

LightGCN was the strongest simple baseline. It captured the core collaborative signal from the user-item graph.

**Lesson:** Keep LightGCN as the main model instead of replacing it.

---

### 3. Recent Popularity

Tested time-decayed item popularity, mainly:

- `recent_pop_180d`
- `recent_pop_365d`

Blended with LightGCN using reciprocal-rank fusion.

**Result:** This was the most useful additional signal.  
`recent_pop_180d` worked best.

**Lesson:** The dataset is strongly time-sensitive. Recent global trends matter a lot.

---

### 4. Metadata and Content

Tested:

- Category-aware popularity
- TF-IDF over item metadata text

**Result:** Small or inconsistent improvements only.

**Lesson:** Metadata contains some signal, but much less than recent popularity.

---

### 5. Sequential Models

Tested:

- Simple sequence transition model
- SASRec Transformer recommender

**Result:** Sequence transition helped slightly in some cases. SASRec did not contribute meaningfully.

**Lesson:** The task appears more trend-driven than sequence-driven. SASRec was not worth including.

---

### 6. Other Collaborative Models

Tested:

- ItemKNN
- Time-decayed LightGCN graph
- Popularity-based negative sampling

**Result:** None improved enough to become part of the final model.

**Lesson:** LightGCN already captured the useful collaborative graph signal well.

---

### 7. Larger LightGCN Models

Tested larger LightGCN backbones:

| Dimension | Layers | Epochs |
|---:|---:|---:|
| 128 | 3 | 160 |
| 256 | 3–4 | 180 |
| 512 | 3–4 | 220 |

**Result:** Larger LightGCN helped. The best model used:

```text
dim = 512
layers = 4
epochs = 220
```

---

## Key Validation and Leaderboard Findings

High recent-popularity alphas performed well locally, but the leaderboard preferred a balanced blend.

| Model | Leaderboard |
|---|---:|
| 256d/4L + recent180, alpha=0.5 | 0.02155 |
| 256d/4L + recent180, alpha=0.6 | 0.02035 |
| 512d/4L + recent180, alpha=0.5 | 0.0236 |

The `alpha=0.5` result suggests that **all-user validation was a better proxy** than target-overlap validation for final leaderboard performance.

---

## Final Model

```text
LightGCN
  dim = 512
  layers = 4
  epochs = 220
  lr = 0.001
  reg = 0.0002
  negative sampling = uniform

Recent popularity
  half-life = 180 days

Blend
  method = RRF
  alpha = 0.5
```

---

## Main Lessons

1. **LightGCN was the right backbone.**
2. **Recent popularity was the best complementary signal.**
3. **Large rerankers and broad ensembles were not necessary.**
4. **SASRec did not help on this dataset.**
5. **Bigger LightGCN improved performance.**
6. **A balanced LightGCN/recency blend generalized better than a more popularity-heavy blend.**

---

## Recommended Next Experiments

Since the current model is already strong, further tests should be focused:

1. Fine-sweep alpha around `0.5`:
   ```text
   0.42, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575
   ```

2. Try nearby recency half-lives:
   ```text
   90, 120, 150, 180, 210, 240, 270
   ```

3. Average multiple LightGCN seeds:
   ```text
   seeds = 42, 43, 44
   ```

4. Test nearby epoch counts:
   ```text
   180, 200, 220, 240, 260
   ```

5. Carefully test a 512d / 5-layer LightGCN to see whether extra depth helps or causes over-smoothing.

---

## Final Takeaway

The winning approach was simple:

```text
Strong LightGCN backbone
+ recent popularity
+ RRF blending
```

The best tested submission used:

```text
512 dimensions
4 layers
220 epochs
180-day recent popularity
alpha = 0.5
```

Leaderboard score:

```text
0.0236
```
