import os
import csv
import ast
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def extract_text(metadata:dict):
    details = ast.literal_eval(metadata['details'])
    brand = f"the {details['Brand']}" if 'Brand' in details else "an unknown"
    description = (
        f"The {metadata['title']} is a {metadata['main_category']} product, made by {brand} company.\n"
        f"{metadata['description'][2:-2]}\n"
        f"{metadata['rating_number']} people rated this product {metadata['average_rating']} on average."
    )
    return description

with open('data/item_meta.csv','r', encoding='utf-8') as file:
    reader = csv.reader(file,delimiter=',')
    header = next(reader,None)
    data = [dict(zip(header,row)) for row in reader]

nl_data = [extract_text(i) for i in data]


# build embeddings
emb_dir = "data/embeddings"
model_name = "sentence-transformers/sentence-t5-base"
device = "cuda"
batch_size = 128

emb_path = os.path.join(emb_dir, "item_embeddings.npy")
item_ids_path = os.path.join(emb_dir, "item_ids.json")
info_path = os.path.join(emb_dir, "embedding_info.json")

item_ids = [x["item_id"] for x in data]


print("Loading sentence encoder...")
model = SentenceTransformer(model_name, device=device)

print("Encoding item texts...")
embeddings = model.encode(
    nl_data,
    batch_size=batch_size,
    convert_to_numpy=True,
    show_progress_bar=True,
)

embeddings = embeddings.astype("float32")

print("Saving embeddings...")
np.save(emb_path, embeddings)

with open(item_ids_path, "w") as f:
    json.dump(item_ids, f, indent=2)

info = {
    "model_name": model_name,
    "num_items": len(item_ids),
    "embedding_dim": embeddings.shape[1],
    "note": "Row i in item_embeddings.npy corresponds to item_idx i.",
}

with open(info_path, "w") as f:
    json.dump(info, f, indent=2)

print("Done.")
print(json.dumps(info, indent=2))