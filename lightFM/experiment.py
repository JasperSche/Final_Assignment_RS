import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import InteractionDataset, BPRDataset
from model import LightFM

TRAIN_PATH      = "data/train.csv"
TEST_PATH       = "data/test.csv"
EMBEDDINGS_PATH = "data/embeddings/item_embeddings.npy"
OUTPUT_PATH     = "lightFM/submission.csv"
SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"

LATENT_DIM  = 64
EPOCHS      = 5
BATCH_SIZE  = 4096
LR          = 1e-3
TOP_K       = 10


def recall_at_k(recommended, relevant, k=10):
    return len(set(recommended) & set(relevant)) / min(len(relevant), k)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset  = InteractionDataset(TRAIN_PATH, TEST_PATH, EMBEDDINGS_PATH)
    bpr_data = BPRDataset(dataset)
    loader   = DataLoader(bpr_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LightFM(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        item_features=dataset.item_features,
        latent_dim=LATENT_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for u, i, j in loader:
            u, i, j = u.to(device), i.to(device), j.to(device)
            loss = model(u, i, j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch}/{EPOCHS}  loss={total_loss / len(loader):.4f}")

    model.eval()

    all_i  = torch.arange(dataset.n_items, device=device)
    item_v = (model.item_latent(all_i) + model.item_feat_proj(model.item_features.to(device))).detach()
    item_b = model.item_bias(all_i).squeeze(1).detach()

    test_users  = sorted(dataset.test["user_id"].unique())
    recalls     = []
    output_rows = []

    for uid in test_users:
        uidx   = dataset.user2idx[uid]
        u_t    = torch.tensor(uidx, device=device)
        u_emb  = model.user_latent(u_t).detach()
        u_b    = model.user_bias(u_t).detach()
        scores = (item_v @ u_emb + u_b + item_b).cpu().numpy()

        top_idx   = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_idx   = top_idx[np.argsort(scores[top_idx])[::-1]]
        rec_items = [dataset.idx2item[i] for i in top_idx]

        if uid in dataset.test_grouped.index:
            recalls.append(recall_at_k(rec_items, dataset.test_grouped[uid], k=TOP_K))


    print(f"\nRecall@{TOP_K}: {np.mean(recalls):.4f}  ({len(recalls)} users)")

    submission_file = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    all_users = submission_file['user_id'].unique()

    output_rows = []
    for uid in all_users:
        uidx   = dataset.user2idx[uid]
        u_t    = torch.tensor(uidx, device=device)
        u_emb  = model.user_latent(u_t).detach()
        u_b    = model.user_bias(u_t).detach()
        scores = (item_v @ u_emb + u_b + item_b).cpu().numpy()

        top_idx   = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_idx   = top_idx[np.argsort(scores[top_idx])[::-1]]
        rec_items = [str(int(dataset.idx2item[i])) for i in top_idx]
        rec_items = ','.join(rec_items)
        output_rows.append(
            {'ID':uid,'user_id':uid,'item_id':rec_items}
        )
    
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
