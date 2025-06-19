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
docs_processed_path = load_artifact_path('docs_processed', file_extension='parquet')
docs_df = pd.read_parquet(docs_processed_path)

def encode_tokens(tokens):
    indices = [word2index.get(t, word2index.get("<UNK>")) for t in tokens]
    input_tensor = torch.tensor(indices, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)
    with torch.no_grad():
        embedding = doc_model((input_tensor, offsets))
    return embedding.squeeze(0).cpu().numpy()

for i, row in docs_df.iterrows():
    key = f"doc:{i}"
    sentence = sentences_df["doc"][i]
    tokens = row["doc"]  # tokenized version
    embedding = encode_tokens(tokens)
    r.hset(key, mapping={
        "text": sentence,
        "embedding": embedding.tobytes()
    })
