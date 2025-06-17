import torch.nn.functional as F
import torch
import tqdm
import json
from utils import load_model_path, init_wandb, get_device
from two_towers import QryTower, DocTower
from dataloader import KeyQueryDataset, getKQDataLoader

EPOCHS = 5
BATCH_SIZE = 3
QUERY_END = 100_000
MARGIN = 0.2
device = get_device()

# Loads embeddings and test out works
init_wandb(5)
ft_embedded_path = load_model_path('fasttext_tensor:latest')
ft_state_dict = torch.load(ft_embedded_path)
num_embeddings, embedding_dim = ft_state_dict["weight"].shape
embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean")
embedding_bag.load_state_dict(ft_state_dict) # Returns Embedding(100001, 300)

# Init Two Towers
qryTower = QryTower(embedding_bag).to(device)
docTower = DocTower(embedding_bag).to(device)

qry_optimizer = torch.optim.Adam(qryTower.parameters(), lr=1e-3)
doc_optimizer = torch.optim.Adam(docTower.parameters(), lr=1e-3)

# May need to change the file path
W2IX_FILE_PATH = 'artifacts/vocab:v0/vocab.json'
with open(W2IX_FILE_PATH) as file:
    w2ix = json.load(file)
dataset = KeyQueryDataset(start=0, end=QUERY_END, word2idx=w2ix)
train_dataloader = getKQDataLoader(dataset, BATCH_SIZE=3)

# Flags to control training
train_qry = True
train_doc = True

for epoch in range(1, EPOCHS+1):
    train_loss = 0.0
    qryTower.train()
    docTower.train()
    for batch in tqdm(train_dataloader):
        feature = feature.to(device)
        query, pos_doc, neg_doc = batch
        query = query.to(device)
        pos_doc = pos_doc.to(device)
        neg_doc = neg_doc.to(device)

        qry_vec = qryTower(query)
        pos_vec = docTower(pos_doc)
        neg_vec = docTower(neg_doc)

        # Detach towers that should not be trained this step
        if not train_qry:
            qry_vec = qry_vec.detach()
        if not train_doc:
            pos_vec = pos_vec.detach()
            neg_vec = neg_vec.detach()

        # Compute similarity scores
        pos_score = torch.nn.functional.cosine_similarity(qry_vec, pos_vec)
        neg_score = torch.nn.functional.cosine_similarity(qry_vec, neg_vec)

        # Difference between the two scores
        dst_dif = pos_score - neg_score
        # Margin
        dst_mrg = torch.tensor(0.2)

        # Loss function
        # Makes the negative doc within the margin
        loss = torch.max(torch.tensor(0.0), dst_mrg - dst_dif)
        # To create the gradients for training

        # Backward
        if train_qry:
            qry_optimizer.zero_grad()
        if train_doc:
            doc_optimizer.zero_grad()
        loss.backward()
        if train_qry:
            qry_optimizer.step()
        if train_doc:
            doc_optimizer.step()

        print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")