import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset:
    def __init__(self, train_path, test_path, embeddings_path):
        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)

        all_users = sorted(train["user_id"].unique())
        all_items = sorted(train["item_id"].unique())

        self.user2idx = {u: i for i, u in enumerate(all_users)}
        self.item2idx = {it: i for i, it in enumerate(all_items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}

        self.n_users = len(all_users)
        self.n_items = len(all_items)

        self.train        = train
        self.test         = test
        self.test_grouped = test.groupby("user_id")["item_id"].apply(list)

        raw_emb = np.load(embeddings_path, allow_pickle=True)
        self.item_features = self._build_item_features(raw_emb)

    def _build_item_features(self, raw_emb):
        emb_dim  = raw_emb.shape[1]
        features = np.zeros((self.n_items, emb_dim), dtype=np.float32)
        for item_id, idx in self.item2idx.items():
            row = item_id - 1
            if 0 <= row < len(raw_emb):
                features[idx] = raw_emb[row]
        return features


class BPRDataset(Dataset):
    def __init__(self, dataset: InteractionDataset):
        self.n_items      = dataset.n_items
        train             = dataset.train
        self.user_indices = train["user_id"].map(dataset.user2idx).values
        self.item_indices = train["item_id"].map(dataset.item2idx).values
        self.seen         = set(zip(self.user_indices.tolist(), self.item_indices.tolist()))

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        u = int(self.user_indices[idx])
        i = int(self.item_indices[idx])
        j = np.random.randint(self.n_items)
        while (u, j) in self.seen:
            j = np.random.randint(self.n_items)
        return u, i, j
