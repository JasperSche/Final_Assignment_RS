# Recent Experiment Summary: Feature Additions After Best LightGCN Baseline

## Starting Point

The previous best model was:

```text
LightGCN 512d / 4 layers / 220 epochs
+ recent popularity, 180-day half-life
+ RRF blend, alpha = 0.5
```

Leaderboard score:

```text
0.0236
```

This was already ranked #1, so the next experiments focused on small, controlled additions rather than large rerankers or new architectures.

---

## Feature Additions Tested

We evaluated five high-priority additions on the validation set:

1. **Dynamic user-dependent alpha**
   - Varies the LightGCN/popularity blend by user history length.
   - Short-history users get more popularity weight.
   - Long-history users get more LightGCN weight.

2. **Recent-window LightGCN**
   - Trains an extra LightGCN only on recent interactions, such as the last 180 or 365 days.
   - Intended to capture recent personalized collaborative structure.

3. **Trend acceleration**
   - Measures whether an item is becoming more popular recently compared with longer-term popularity.
   - Main variants: `trend90v365` and `trend120v365`.

4. **Multi-seed LightGCN averaging**
   - Trains the same LightGCN with multiple seeds and combines rankings using RRF.
   - Tested mainly with seeds `42,43,44`, then later `42,43,44,45,46`.

5. **Category-conditioned recent popularity**
   - Uses item metadata to find each user's dominant category.
   - Boosts recently popular items within that category.

---

## Validation Findings

The most promising validation result was:

```text
3-seed LightGCN
+ recent_pop_180d
+ trend120v365
```

Key observations:

- **Trend acceleration looked strongest on validation.**
- `trend120v365` and `trend90v365` were the best trend variants.
- **3-seed LightGCN + recent popularity** improved over single-seed in validation.
- **Dynamic alpha** produced some promising target-overlap validation scores but weaker all-user validation.
- **Recent-window LightGCN** did not clearly outperform the trend models.
- **Category-conditioned popularity** helped target validation somewhat but was not strong enough overall.

Based on validation, the main candidates for submission were:

```text
3-seed LightGCN + recent180 alpha=0.5
3-seed LightGCN + recent180 alpha=0.5 + trend120v365
3-seed LightGCN + recent180 alpha=0.5 + trend90v365
dynamic-alpha LightGCN + recent180
```

---

## Submission Results

The 3-seed submissions produced:

| Submission | Leaderboard Recall@10 |
|---|---:|
| 3-seed + recent180 alpha=0.5 | **0.02451** |
| 3-seed + recent180 alpha=0.5 + trend120v365 gamma=0.3 | **0.02451** |
| 3-seed + recent180 alpha=0.5 + trend120v365 gamma=0.1 | **0.02451** |
| 3-seed + recent180 alpha=0.5 + trend90v365 gamma=0.3 | **0.02451** |
| 3-seed dynamic alpha | 0.02397 |

Main interpretation:

```text
3-seed LightGCN + recent180 alpha=0.5 is the new strongest confirmed model.
```

The trend variants tied the fixed model on the leaderboard. This likely means trend changed too few top-10 recommendations, or changed them without changing the number of public hits.

---

## 5-Seed Experiment

We then tested a larger seed ensemble:

```text
seeds = 42,43,44,45,46
```

The fixed 5-seed model scored:

```text
0.02318
```

This was significantly worse than the 3-seed model:

```text
3-seed: 0.02451
5-seed: 0.02318
```

Conclusion:

```text
More seeds did not help.
```

The 5-seed average likely smoothed away useful ranking sharpness. We decided not to spend more submissions on 5-seed variants.

---

## Current Best Model

The current best confirmed model is:

```text
LightGCN
  dim = 512
  layers = 4
  epochs = 220
  lr = 0.001
  reg = 0.0002
  seeds = 42,43,44

Recent popularity
  half-life = 180 days

Blend
  method = RRF
  alpha = 0.5
```

Leaderboard score:

```text
0.02451
```

