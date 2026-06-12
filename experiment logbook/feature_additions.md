## 1. Dynamic user-dependent alpha

Instead of using the same LightGCN/popularity blend for every user, alpha depends on the user’s history length.

Base blend:

```text
score = (1 - alpha) * LightGCN + alpha * recent_popularity
```

Dynamic rule examples:

```text
history length <= 3       alpha = 0.65 or 0.75
history length <= 6       alpha = 0.55 or 0.65
history length <= 12      alpha = 0.50 or 0.55
history length > 12       alpha = 0.40 or 0.45
```

Interpretation:

```text
short-history users -> more popularity
long-history users  -> more LightGCN personalization
```

---

## 2. Recent-window LightGCN

Train a second LightGCN using only recent interactions.

Example:

```text
Full LightGCN:
  trained on all train_context interactions

Recent LightGCN:
  trained only on interactions from the last 180 or 365 days
```

Then blend:

```text
score = LightGCN_full + recent_popularity + gamma * LightGCN_recent
```

In the script:

```text
gamma ∈ {0.05, 0.1, 0.15, 0.2, 0.3}
```

Purpose:

```text
capture personalized collaborative patterns that are recent, not just globally popular
```

---

## 3. Trend acceleration score

This measures whether an item is becoming more popular recently compared with its longer-term popularity.

For example:

```text
trend_30v365
trend_60v365
trend_90v365
trend_120v365
```

Calculation:

```text
short = time-decayed item count with short half-life
long  = time-decayed item count with long half-life

trend = log(1 + short) - log(1 + long) + 0.15 * log(1 + short)
```

Then blend:

```text
score = LightGCN + recent_popularity + gamma * trend_score
```

Purpose:

```text
boost items that are rising now, not merely items that were popular historically
```

---

## 4. Multi-seed LightGCN averaging

Train the same LightGCN architecture multiple times with different random seeds.

Example:

```text
seed 42
seed 43
seed 44
```

Each seed produces a LightGCN ranking. The rankings are combined using RRF:

```text
LightGCN_multiseed = average(RRF(seed_42), RRF(seed_43), RRF(seed_44))
```

Then blend with recent popularity:

```text
score = (1 - alpha) * LightGCN_multiseed + alpha * recent_popularity
```

Purpose:

```text
reduce random seed variance and get a more stable ranking
```

---

## 5. Category-conditioned recent popularity

Use item metadata to estimate what categories a user prefers.

Steps:

```text
1. Map each item to its main_category.
2. For each user, find their dominant historical category.
3. Compute recent popularity within each category.
4. Score items higher if they are recently popular in the user’s dominant category.
```

In the script:

```text
category_score = category_recent_popularity + 0.20 * global_recent_popularity
```

Then blend:

```text
score = LightGCN + recent_popularity + gamma * category_score
```

Purpose:

```text
recommend items that are both currently popular and aligned with the user's preferred category
```

---

## Common blending method

Most additions are combined with **RRF**, not raw scores.

For each model/component:

```text
rank 1 item -> 1 / (60 + 1)
rank 2 item -> 1 / (60 + 2)
rank n item -> 1 / (60 + n)
```

This avoids problems where LightGCN dot products, popularity counts, and trend scores have different scales.
