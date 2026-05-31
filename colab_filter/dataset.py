import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class InteractionDataset:
    def __init__(self, train_path, test_path, recency_scale=100.0):
        self.recency_scale = recency_scale

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        all_users = sorted(train["user_id"].unique())
        all_items = sorted(train["item_id"].unique())

        self.user2idx = {u: i for i, u in enumerate(all_users)}
        self.item2idx = {it: i for i, it in enumerate(all_items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}
        self.idx2user = {i: u for u, i in self.user2idx.items()}

        self.n_users = len(all_users)
        self.n_items = len(all_items)

        self.train = train
        self.test = test

        self.user_item = self._build_matrix(train)
        self.test_grouped = test.groupby("user_id")["item_id"].apply(list)

    def _build_matrix(self, df):
        rows = df["user_id"].map(self.user2idx).values
        cols = df["item_id"].map(self.item2idx).values

        ts = df["timestamp"].values.astype(np.float32)
        t_min, t_max = ts.min(), ts.max()
        conf = 1.0 + self.recency_scale * (ts - t_min) / (t_max - t_min)

        return csr_matrix(
            (conf, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )