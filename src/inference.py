import os
import numpy as np
import redis
import torch
from two_towers import QryTower, DocTower
from utils import get_device, load_model_path, init_wandb, load_artifact_path
import json
from fasttext.FastText import tokenize
import pandas as pd

device = get_device()
QUERY_MODEL_VERSION = "v68"
DOC_MODEL_VERSION = "v68"

init_wandb()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# --- Set up vocab and all_docs artifacts needed later ---
vocab_path = load_model_path('vocab:latest')
with open(vocab_path) as f:
    word2index = json.load(f)

def get_all_docs():
    docs_path = load_artifact_path('docs', file_extension='parquet')
    docs = pd.read_parquet(docs_path)
    docs_processed_path = load_artifact_path('docs_processed', file_extension='parquet')
    docs_processed = pd.read_parquet(docs_processed_path)

    return {
        "tokenized": docs_processed["doc"],
        "sentences": docs["doc"]
    }
all_docs = get_all_docs()

# --- Initialising model ---
def _init_models():
    ft_embedded_path = load_model_path('fasttext_tensor:latest')
    ft_state_dict = torch.load(ft_embedded_path, map_location=device)
    num_embeddings, embedding_dim = ft_state_dict["weight"].shape
    embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean")
    embedding_bag.load_state_dict(ft_state_dict)

    query_model_path = load_model_path(f'query_model:{QUERY_MODEL_VERSION}')
    query_model = QryTower(embedding_bag).to(device)
    query_model.load_state_dict(torch.load(query_model_path, map_location=device))
    query_model.eval()

    doc_model_path = load_model_path(f'doc_model:{DOC_MODEL_VERSION}')
    doc_model = DocTower(embedding_bag).to(device)
    doc_model.load_state_dict(torch.load(doc_model_path, map_location=device))
    doc_model.eval()
    return doc_model, query_model

doc_model, query_model = _init_models()

def search_query(query: str, num_doc=5):
    query_tokens = tokenize(query)
    query_indices = [word2index.get(t, word2index.get("<UNK>")) for t in query_tokens]
    input_tensor = torch.tensor(query_indices, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)

    with torch.no_grad():
        query_vec = query_model((input_tensor, offsets)).squeeze(0).cpu().numpy()

    base64_vec = query_vec.astype(np.float32).tobytes()

    redis_query = [
        "FT.SEARCH",
        "doc_idx",
        f"*=>[KNN {num_doc} @embedding $vec AS __embedding_score]",
        "RETURN", "3", "text", "embedding", "__embedding_score",
        "SORTBY", "__embedding_score",
        "LIMIT", "0", str(num_doc),
        "PARAMS", "2", "vec", base64_vec,
        "DIALECT", "2",
    ]

    res = r.execute_command(*redis_query)

    # Parse Redis results
    results = []
    for i in range(1, len(res), 2):  # skip total count
        doc = res[i + 1][3]  # get the 'text' field
        score = 1 - float(res[i + 1][1]) # __embedding_score from above query, to get cosine need to do "1- __embedding_score" since it returns distances
        results.append({
            "doc": doc.decode("utf-8") if isinstance(doc, bytes) else doc,
            "score": round(score, 4)
        })

    sorted_results = [
        {**item, "rank": idx + 1}
        for idx, item in enumerate(sorted(results, key=lambda x: x["score"], reverse=True))
    ]

    return sorted_results
