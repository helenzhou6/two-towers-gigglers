import os
import torch
import redis
from utils import load_model_path, get_device, init_wandb, load_artifact_path
from two_towers import DocTower
import pandas as pd
import json

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

init_wandb()
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

device = get_device()

# Load model
ft_path = load_model_path('fasttext_tensor:latest')
ft_state_dict = torch.load(ft_path, map_location=device)
embedding_bag = torch.nn.EmbeddingBag.from_pretrained(ft_state_dict['weight'], mode="mean")
doc_model = DocTower(embedding_bag).to(device)
doc_model.load_state_dict(torch.load(load_model_path("doc_model:latest"), map_location=device))
doc_model.eval()

# Load vocab and data
vocab_path = load_model_path('vocab:latest')
with open(vocab_path) as f:
    word2index = json.load(f)

docs_path = load_artifact_path('docs', file_extension='parquet')
sentences_df = pd.read_parquet(docs_path)
docs_indices_path = load_artifact_path('docs_processed', file_extension='parquet')
docs_indices_df = pd.read_parquet(docs_indices_path)

# TODO: To remove the commented out code - running so can run on Helen's laptop XD
# sentences_df = pd.read_parquet('data/docs.parquet').head(10)
# docs_indices_df = pd.read_parquet('data/docs_processed.parquet').head(10)

print("Artifact load finished, putting vectors in redis vector database...")

def encode_tokens(indices):
    input_tensor = torch.tensor(indices, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)
    with torch.no_grad():
        embedding = doc_model((input_tensor, offsets))
    return embedding.squeeze(0).cpu().numpy()

for i, row in docs_indices_df.iterrows():
    key = f"doc:{i}"
    sentence = sentences_df["doc"][i]
    indices = row["doc"]  # tokenized version
    embedding = encode_tokens(indices)
    r.hset(key, mapping={
        "text": sentence,
        "embedding": embedding.tobytes()
    })
