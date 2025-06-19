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

ft_embedded_path = load_model_path('fasttext_tensor:latest')
ft_state_dict = torch.load(ft_embedded_path, map_location=device)
num_embeddings, embedding_dim = ft_state_dict["weight"].shape
embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean")
embedding_bag.load_state_dict(ft_state_dict)

vocab_path = load_model_path('vocab:latest')
with open(vocab_path) as f:
    word2index = json.load(f)

query_model_path = load_model_path(f'query_model:{QUERY_MODEL_VERSION}')
query_model = QryTower(embedding_bag).to(device)
query_model.load_state_dict(torch.load(query_model_path, map_location=device))
query_model.eval()

doc_model_path = load_model_path(f'doc_model:{DOC_MODEL_VERSION}')
doc_model = DocTower(embedding_bag).to(device)
doc_model.load_state_dict(torch.load(doc_model_path, map_location=device))
doc_model.eval()

def prepare_embeddingbag_inputs(tokens_indices):
    input_tensor = torch.tensor(tokens_indices, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)
    return (input_tensor, offsets)

docs_path = load_artifact_path('docs', file_extension='parquet')
docs = pd.read_parquet(docs_path)
docs_processed_path = load_artifact_path('docs_processed', file_extension='parquet')
docs_processed = pd.read_parquet(docs_processed_path)

all_docs = {
    "tokenized": docs_processed["doc"],
    "sentences": docs["doc"]
}

def get_top_docs(query_embedding, num_doc=2):
    print("Starting to search for top docs...")
    doc_embeddings = []
    for tokens in all_docs["tokenized"]:
        doc_tensor = doc_model(prepare_embeddingbag_inputs(tokens))
        doc_embeddings.append(doc_tensor.squeeze(0))  # Ensure shape [300]

    db = torch.stack(doc_embeddings)  # Now shape [num_docs, 300]
    query_embedding = query_embedding.squeeze(0)

    res = torch.nn.functional.cosine_similarity(db, query_embedding, dim=1)
    top_scr, top_idx = torch.topk(res, k=num_doc)

    return [
        {
            "rank": rank + 1,
            "doc": all_docs["sentences"][i.item()],
            "score": round(s.item(), 4)
        }
        for rank, (s, i) in enumerate(zip(top_scr, top_idx))
    ]


def search_query(query: str, num_doc=5):
    query_tokens = tokenize(query)

    query_indices = [word2index[t] for t in query_tokens if t in word2index]
    unknown_index = word2index.get("<UNK>")
    if not query_indices:
        query_indices = [unknown_index]

    query_input = prepare_embeddingbag_inputs(query_indices)

    with torch.no_grad():
        query_embedding = query_model(query_input)

    return get_top_docs(query_embedding, num_doc)
