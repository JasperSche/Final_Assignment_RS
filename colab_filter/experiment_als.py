import numpy as np
import pandas as pd

from dataset import InteractionDataset
from model import ALSRecommender

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "colab_filter/submission.csv"

RECENCY_SCALE = 100.0
FACTORS = 256
ITERATIONS = 30
REGULARIZATION = 0.01
TOP_K = 10


def recall_at_k(recommended, relevant, k=10):
    return len(set(recommended) & set(relevant)) / min(len(relevant), k)


def main(make_submission = True):
    dataset = InteractionDataset(TRAIN_PATH, TEST_PATH, recency_scale=RECENCY_SCALE)

    model = ALSRecommender(
        factors=FACTORS,
        iterations=ITERATIONS,
        regularization=REGULARIZATION,
    )
    model.fit(dataset.user_item)

    test_users = sorted(dataset.test["user_id"].unique())
    recalls = []    

    for uid in test_users:
        uidx = dataset.user2idx[uid]
        rec_indices = model.recommend(uidx, n=TOP_K)
        rec_items = [dataset.idx2item[i] for i in rec_indices]

        if uid in dataset.test_grouped.index:
            true_items = dataset.test_grouped[uid]
            recalls.append(recall_at_k(rec_items, true_items, k=TOP_K))

    print(f"Test set Recall@{TOP_K}: {np.mean(recalls):.4f}  ({len(recalls)} users)")


    submission_file = pd.read_csv('data/sample_submission.csv')
    all_users = submission_file['user_id'].unique()

    output_rows = []
    for uid in all_users:
        uidx = dataset.user2idx[uid]
        rec_indices = model.recommend(uidx, n=TOP_K)
        rec_items = [str(dataset.idx2item[i]) for i in rec_indices]
        rec_items = ','.join(rec_items)
        output_rows.append(
            {'ID':uid,'user_id':uid,'item_id':rec_items}
        )

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()