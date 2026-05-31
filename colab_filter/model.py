import numpy as np
import implicit


class ALSRecommender:
    def __init__(self, factors=256, iterations=30, regularization=0.01, random_state=42):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=random_state,
        )

    def fit(self, user_item):
        self.model.fit(user_item)
        self.user_item = user_item

    def recommend(self, user_idx, n=10):
        recs, _ = self.model.recommend(
            user_idx,
            self.user_item[user_idx],
            N=n,
            filter_already_liked_items=False,
        )
        return recs

    def recommend_batch(self, user_indices, n=10):
        return [self.recommend(uidx, n) for uidx in user_indices]