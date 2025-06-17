import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from utils import load_model_path, init_wandb, get_device, save_model
from two_towers import QryTower, DocTower
from dataloader import KeyQueryDataset, collate_fn_emb_bag, getKQDataLoader

LEARNING_RATE = 0.01
EPOCHS = 5
BATCH_SIZE = 32
QUERY_END = 10000
MARGIN = torch.tensor(0.2)
device = get_device()

# Loads embeddings and test out works
init_wandb(LEARNING_RATE, EPOCHS)
ft_embedded_path = load_model_path('fasttext_tensor:latest')
ft_state_dict = torch.load(ft_embedded_path)
num_embeddings, embedding_dim = ft_state_dict["weight"].shape
embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean")
embedding_bag.load_state_dict(ft_state_dict) # Returns Embedding(100001, 300)

# Init Two Towers
query_model = QryTower(embedding_bag).to(device)
doc_model = DocTower(embedding_bag).to(device)

qry_optimizer = torch.optim.Adam(query_model.parameters(), lr=LEARNING_RATE)
doc_optimizer = torch.optim.Adam(doc_model.parameters(), lr=LEARNING_RATE)

# May need to change the file path
vocab_path = load_model_path('vocab:latest')
with open(vocab_path) as file:
    w2ix = json.load(file)
dataset = KeyQueryDataset(start=0, end=QUERY_END, word2idx=w2ix)
train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_emb_bag)

for epoch in range(0, EPOCHS):
    train_loss = 0.0
    query_model.train()
    doc_model.train()
    for batch in tqdm(train_dataloader):
        query, pos_doc, neg_doc = batch
        query = (query[0].to(device), query[1].to(device))   # inputs, offsets
        pos_doc = (pos_doc[0].to(device), pos_doc[1].to(device))
        neg_doc = (neg_doc[0].to(device), neg_doc[1].to(device))

        qry_vec = query_model(query)
        pos_vec = doc_model(pos_doc)
        neg_vec = doc_model(neg_doc)

        # Compute similarity scores
        pos_score = torch.nn.functional.cosine_similarity(qry_vec, pos_vec)
        neg_score = torch.nn.functional.cosine_similarity(qry_vec, neg_vec)

        # Difference between the two scores
        distance_scores = pos_score - neg_score
        loss = torch.max(torch.tensor(0.0), MARGIN - distance_scores)

        # Backward
        qry_optimizer.zero_grad()
        doc_optimizer.zero_grad()

        # loss from whole batch average
        loss = loss.mean()
        loss.backward()

        for name, param in query_model.named_parameters():
          if param.grad is None:
              print(f"No grad for {name}")
          else:
              print(f"Grad for {name}: {param.grad.norm()}")

        qry_optimizer.step()
        doc_optimizer.step()
        
        print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

torch.save(query_model.state_dict(), 'data/query_model.pt')
save_model('query_model', 'The trained model for our queries')

torch.save(doc_model.state_dict(), 'data/doc_model.pt')
save_model('doc_model', 'The trained model for our documents')
