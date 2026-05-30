import torch
import torch.nn as nn
import torch.nn.functional as F


class LightFM(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_item_features: int, embedding_dim: int = 32):
        super(LightFM,self).__init__()
 
        self.user_latent = nn.Embedding(n_users, embedding_dim)
        self.user_bias   = nn.Embedding(n_users, 1)
 
        self.item_id_latent = nn.Embedding(n_items, embedding_dim)
        self.item_id_bias   = nn.Embedding(n_items, 1)
 
        self.item_feat_latent = nn.Embedding(n_item_features, embedding_dim)
        self.item_feat_bias   = nn.Embedding(n_item_features, 1)
 
        self._init_weights()
 
    def _init_weights(self):
        for emb in [self.user_latent, self.item_id_latent, self.item_feat_latent,        ]:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in [self.user_bias, self.item_id_bias, self.item_feat_bias]:
            nn.init.zeros_(emb.weight)
 
    def get_item_representation(self,item_ids: torch.Tensor,item_feat_ids: torch.Tensor, item_feat_mask: torch.Tensor | None = None,):
        id_latent = self.item_id_latent(item_ids)  
        id_bias   = self.item_id_bias(item_ids)     
 
        feat_latent = self.item_feat_latent(item_feat_ids)  
        feat_bias   = self.item_feat_bias(item_feat_ids)    
 
        if item_feat_mask is not None:
            mask_l = item_feat_mask.unsqueeze(-1).float()   
            feat_latent = feat_latent * mask_l
            feat_bias   = feat_bias   * mask_l
 
        item_latent = id_latent + feat_latent.sum(dim=1)    
        item_bias   = id_bias   + feat_bias.sum(dim=1)      
 
        return item_latent, item_bias
 
    def forward(self, user_ids:torch.Tensor, item_ids:torch.Tensor, item_feat_ids:torch.Tensor, item_feat_mask:torch.Tensor | None = None) -> torch.Tensor:
        u_latent = self.user_latent(user_ids)    
        u_bias   = self.user_bias(user_ids)       
 
        i_latent, i_bias = self.get_item_representation(
            item_ids, item_feat_ids, item_feat_mask
        )
 
        dot   = (u_latent * i_latent).sum(dim=-1, keepdim=True)  
        score = dot + u_bias + i_bias                             
        return score.squeeze(-1)                                  
 
 
def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()
 
 

if __name__ == "__main__":
    # TODO: override to get test on the actual data
    torch.manual_seed(42)
 
    N_USERS         = 1_000
    N_ITEMS         = 5_000
    N_ITEM_FEATURES = 200    
    EMB_DIM         = 32
    N_FEAT_PER_ITEM = 5      
    BATCH_SIZE      = 256
    LR              = 1e-3
    N_EPOCHS        = 10
 
    model = LightFM(N_USERS, N_ITEMS, N_ITEM_FEATURES, EMB_DIM)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
 
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")
 
    for epoch in range(1, N_EPOCHS + 1):
        user_ids = torch.randint(0, N_USERS, (BATCH_SIZE,))
 
        pos_item_ids   = torch.randint(0, N_ITEMS, (BATCH_SIZE,))
        pos_feat_ids   = torch.randint(0, N_ITEM_FEATURES, (BATCH_SIZE, N_FEAT_PER_ITEM))
        pos_feat_mask  = torch.ones(BATCH_SIZE, N_FEAT_PER_ITEM)
 
        neg_item_ids   = torch.randint(0, N_ITEMS, (BATCH_SIZE,))
        neg_feat_ids   = torch.randint(0, N_ITEM_FEATURES, (BATCH_SIZE, N_FEAT_PER_ITEM))
        neg_feat_mask  = torch.ones(BATCH_SIZE, N_FEAT_PER_ITEM)
 
        pos_scores = model(user_ids, pos_item_ids, pos_feat_ids, pos_feat_mask)
        neg_scores = model(user_ids, neg_item_ids, neg_feat_ids, neg_feat_mask)
 
        loss = bpr_loss(pos_scores, neg_scores)
 
        opt.zero_grad()
        loss.backward()
        opt.step()
 
        print(f"Epoch {epoch:>2}  |  BPR loss: {loss.item():.4f}")
 
    model.eval()
    with torch.no_grad():
        scores = model(
            user_ids      = torch.tensor([0, 1, 2]),
            item_ids      = torch.tensor([10, 20, 30]),
            item_feat_ids = torch.randint(0, N_ITEM_FEATURES, (3, N_FEAT_PER_ITEM)),
        )
    print(f"\nSample inference scores: {scores.tolist()}")
 
