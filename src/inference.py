import torch
from two_towers import QryTower, DocTower
from utils import get_device, load_model_path, init_wandb, load_artifact_path
import json
from fasttext.FastText import tokenize
import pandas as pd

device = get_device()

init_wandb()

ft_embedded_path = load_model_path('fasttext_tensor:latest')
ft_state_dict = torch.load(ft_embedded_path, map_location=device)
num_embeddings, embedding_dim = ft_state_dict["weight"].shape
embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean")
embedding_bag.load_state_dict(ft_state_dict)

vocab_path = load_model_path('vocab:latest')
with open(vocab_path) as f:
    word2index = json.load(f)

query_model_path = load_model_path('query_model:latest')
query_model = QryTower(embedding_bag).to(device)
query_model.load_state_dict(torch.load(query_model_path, map_location=device))
query_model.eval()

doc_model_path = load_model_path('doc_model:latest')
doc_model = DocTower(embedding_bag).to(device)
doc_model.load_state_dict(torch.load(doc_model_path, map_location=device))
doc_model.eval()

def prepare_embeddingbag_inputs(tokens, word2index):
    indices = [word2index[t] for t in tokens if t in word2index]
    unknown_index = word2index.get("<UNK>", len(tokens))
    if not indices:
        indices = [unknown_index]
    input_tensor = torch.tensor(indices, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)
    return (input_tensor, offsets)

docs_path = load_artifact_path('docs', file_extension='parquet')
docs = pd.read_parquet(docs_path)
docs_processed_path = load_artifact_path('docs_processed', file_extension='parquet')
docs_processed = pd.read_parquet(docs_processed_path)
all_docs = pd.concat([
    docs["doc"].rename("sentences"),
    docs_processed["doc"].rename("tokenized")
], axis=1)

def get_top_docs(query_embedding, num_doc=2):
    # Create a copy to avoid modifying the original DataFrame
    df = all_docs.copy()
    # Compute cosine similarity for each document
    df["similarity"] = df["tokenized"].apply(
        lambda doc_tokens: torch.nn.functional.cosine_similarity(
            query_embedding,
            doc_model(prepare_embeddingbag_inputs(doc_tokens, word2index))
        ).item()
    )
    top_docs = (
        df.sort_values(by="similarity", ascending=False)
          .head(num_doc)
          .reset_index(drop=True)
    )
    top_docs["rank"] = top_docs.index + 1
    top_docs.loc[:, "title"] = top_docs["sentences"]
    top_docs["score"] = top_docs["similarity"].round(4)
    return top_docs[["rank", "title", "score"]].to_dict(orient="records")

def search_query(query: str, num_doc=2):
    query_tokens = tokenize(query)
    query_input = prepare_embeddingbag_inputs(query_tokens, word2index)

    with torch.no_grad():
        query_embedding = query_model(query_input)

    return get_top_docs(query_embedding, num_doc)

results = search_query("home pickled eggs causing botulism at room temperature")
for result in results:
    print(result)