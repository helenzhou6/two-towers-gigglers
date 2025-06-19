import torch
from two_towers import QryTower, DocTower
from utils import get_device, load_model_path, init_wandb
import json
from fasttext.FastText import tokenize

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
    idxs = [word2index[t] for t in tokens if t in word2index]
    unknown_index = word2index.get("<UNK>", 0)
    if not idxs:
        idxs = [unknown_index]
    input_tensor = torch.tensor(idxs, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)
    return (input_tensor, offsets)

# TODO: Instead of the below, set up with the document tower

# ds = dataset.Triplets(w2v.emb, words_to_ids)
# db = torch.stack([doc_model(ds.to_emb(ds.docs[k]).to(dev)) for k in ds.d_keys])
candidate_docs = [
    ["example", "document", "text"],
    ["another", "sample", "doc"],
    ["yet", "another", "document", "example"],
]
def get_top_docs(query_embedding, num_doc = 2):
    similarities = [
        torch.nn.functional.cosine_similarity(query_embedding, doc_model(prepare_embeddingbag_inputs(doc_tokens, word2index))).item()
        for doc_tokens in candidate_docs
    ]
    top_indices = sorted(range(len(similarities)), key=similarities.__getitem__, reverse=True)[:num_doc]
    return [
        {
            "rank": rank,
            "doc": " ".join(candidate_docs[idx]),
            "score": round(similarities[idx], 4)
        }
        for rank, idx in enumerate(top_indices, 1)
    ]

def search_query(query: str, num_doc=2):
    query_tokens = tokenize(query)
    query_input = prepare_embeddingbag_inputs(query_tokens, word2index)

    with torch.no_grad():
        query_embedding = query_model(query_input)

    return get_top_docs(query_embedding, 2)

results = search_query("sample doc")
for result in results:
    print(result)