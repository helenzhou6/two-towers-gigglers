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
QUERY_MODEL_VERSION = "latest"
DOC_MODEL_VERSION = "latest"

init_wandb()

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

# --- Util functions needed to get top docs ---
def _prepare_embeddingbag_inputs(tokens_indices):
    input_tensor = torch.tensor(tokens_indices, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)
    return (input_tensor, offsets)

def _query_to_embedding(query):
    query_tokens = tokenize(query)
    query_indices = [word2index[t] for t in query_tokens if t in word2index]
    unknown_index = word2index.get("<UNK>")
    if not query_indices:
        query_indices = [unknown_index]

    query_input = _prepare_embeddingbag_inputs(query_indices)
    with torch.no_grad():
        query_embedding = query_model(query_input)
    return query_embedding.squeeze(0)

def _get_doc_stack():
    doc_embeddings = []
    for tokens in all_docs["tokenized"]:
        with torch.no_grad():
            doc_tensor = doc_model(_prepare_embeddingbag_inputs(tokens))
            doc_embeddings.append(doc_tensor.squeeze(0))  # Ensure shape [300]
    return torch.stack(doc_embeddings)  # Now shape [num_docs, 300]

def search_query(query: str, num_doc=5):
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    query_tokens = tokenize(query)
    query_indices = [word2index.get(t, word2index.get("<UNK>")) for t in query_tokens]
    input_tensor = torch.tensor(query_indices, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)

    with torch.no_grad():
        query_vec = query_model((input_tensor, offsets)).squeeze(0).cpu().numpy()

    base64_vec = query_vec.astype(np.float32).tobytes()

    redis_query = f"""FT.SEARCH doc_idx "*"
        RETURN 2 text embedding
        PARAMS 2 vec "{base64_vec}"
        DIALECT 2
        SORTBY __embedding_score
        LIMIT 0 {num_doc}
        VECTOR => {{
            "TYPE": "FLOAT32",
            "DIM": 300,
            "DISTANCE_METRIC": "COSINE",
            "VECTOR": $vec,
            "PROPERTY": "embedding"
        }}"""

    res = r.execute_command(*redis_query.split())

    # Parse Redis results
    results = []
    for i in range(1, len(res), 2):  # skip total count
        doc = res[i + 1][1]  # get the 'text' field
        score = 1.0  # Redis doesn't return score by default
        results.append({
            "rank": len(results) + 1,
            "doc": doc.decode("utf-8") if isinstance(doc, bytes) else doc,
            "score": round(score, 4)
        })

    return results

# # --- The function to run! See test folder to test run this ---
# def search_query(query: str, num_doc=5):
#     query_embedding = _query_to_embedding(query)
#     print("Starting to search for top docs...")
#     doc_stack = _get_doc_stack()
#     res = torch.nn.functional.cosine_similarity(doc_stack, query_embedding, dim=1)
#     top_scr, top_idx = torch.topk(res, k=num_doc)

#     return [
#         {
#             "rank": rank + 1,
#             "doc": all_docs["sentences"][i.item()],
#             "score": round(s.item(), 4)
#         }
#         for rank, (s, i) in enumerate(zip(top_scr, top_idx))
#     ]
