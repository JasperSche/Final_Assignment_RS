import torch
import torch.nn as nn
import torch.nn.functional as F


class LightFM(nn.Module):
    def __init__(self, n_users, n_items, item_features, latent_dim=64):
        super().__init__()
        emb_dim = item_features.shape[1]

        self.user_latent    = nn.Embedding(n_users, latent_dim)
        self.item_latent    = nn.Embedding(n_items, latent_dim)
        self.user_bias      = nn.Embedding(n_users, 1)
        self.item_bias      = nn.Embedding(n_items, 1)
        self.item_feat_proj = nn.Linear(emb_dim, latent_dim, bias=False)

        self.register_buffer("item_features", torch.tensor(item_features))

        nn.init.normal_(self.user_latent.weight,    std=0.01)
        nn.init.normal_(self.item_latent.weight,    std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.item_feat_proj.weight, std=0.01)

    def _item_repr(self, item_idx):
        return self.item_latent(item_idx) + self.item_feat_proj(self.item_features[item_idx])

    def forward(self, user_idx, pos_idx, neg_idx):
        u  = self.user_latent(user_idx)
        ub = self.user_bias(user_idx).squeeze(1)
        pi = self._item_repr(pos_idx)
        pb = self.item_bias(pos_idx).squeeze(1)
        ni = self._item_repr(neg_idx)
        nb = self.item_bias(neg_idx).squeeze(1)
        pos_score = (u * pi).sum(1) + ub + pb
        neg_score = (u * ni).sum(1) + ub + nb
        return -F.logsigmoid(pos_score - neg_score).mean()

    @torch.no_grad()
    def all_item_scores(self, user_idx):
        u      = self.user_latent(user_idx)
        ub     = self.user_bias(user_idx)
        all_i  = torch.arange(self.item_latent.num_embeddings, device=u.device)
        item_v = self._item_repr(all_i)
        ib     = self.item_bias(all_i).squeeze(1)
        return item_v @ u + ub + ib
