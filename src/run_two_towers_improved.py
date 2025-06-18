import torch
import wandb
import json
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_model_path, init_wandb, get_device, save_model
from two_towers import QryTower, DocTower
from dataloader import KeyQueryDataset, collate_fn_emb_bag, getKQDataLoader

LEARNING_RATE = 1e-3
EPOCHS = 5
BATCH_SIZE = 64
QUERY_END = 5_000_000
MARGIN = torch.tensor(0.2)
device = get_device()

print('Starting... Two Towers Document Search Training\n\n')

print(f'Using device {device}.')

# Loads embeddings and test out works
init_wandb(LEARNING_RATE, EPOCHS)
ft_embedded_path = load_model_path('fasttext_tensor:latest')
ft_state_dict = torch.load(ft_embedded_path, map_location=device)
num_embeddings, embedding_dim = ft_state_dict["weight"].shape

embedding_bag_doc = embedding_bag_qry = torch.nn.EmbeddingBag.from_pretrained(
    embeddings=ft_state_dict["weight"],
    freeze=True,
    padding_idx=ft_state_dict.get('padding_idx', None),
    mode='mean',
    sparse=ft_state_dict.get('sparse', False)
).to(device)

embedding_bag_query = torch.nn.EmbeddingBag.from_pretrained(
    embeddings=ft_state_dict["weight"],
    freeze=True,
    padding_idx=ft_state_dict.get('padding_idx', None),
    mode='mean',
    sparse=ft_state_dict.get('sparse', False)
).to(device)

# Init Two Towers
query_model = QryTower(embedding_bag_query).to(device)
doc_model = DocTower(embedding_bag_doc).to(device)

wandb.watch(query_model, log="all", log_freq=100)
wandb.watch(doc_model, log="all", log_freq=100)

optimizer = torch.optim.Adam(
    list(query_model.parameters()) + list(doc_model.parameters()),
    lr=LEARNING_RATE
)

# ── 4) Use the built-in triplet loss with cosine distance ───────────────────────
criterion = torch.nn.TripletMarginWithDistanceLoss(
    margin=MARGIN,
    distance_function=lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, dim=1),
    reduction='mean'
)

# May need to change the file path
vocab_path = load_model_path('vocab:latest')
with open(vocab_path) as file:
    w2ix = json.load(file)

dataset = KeyQueryDataset(start=0, end=QUERY_END, word2idx=w2ix)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_emb_bag)

for epoch in range(0, EPOCHS):
    query_model.train()
    doc_model.train()
    total_loss = 0.0
    for (q_flat, q_off), (pos_flat, pos_off), (neg_flat, neg_off) in tqdm(
        dataloader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    ):
        # move to device
        q_flat, q_off = q_flat.to(device), q_off.to(device)
        pos_flat, pos_off = pos_flat.to(device), pos_off.to(device)
        neg_flat, neg_off = neg_flat.to(device), neg_off.to(device)

        # forward
        optimizer.zero_grad()

        q_vec = query_model((q_flat, q_off))
        pos_vec = doc_model((pos_flat, pos_off))
        neg_vec = doc_model((neg_flat, neg_off))

        # compute scalar loss
        loss = criterion(q_vec, pos_vec, neg_vec)
        
        # backward + step
        loss.backward()
        optimizer.step()

        if os.getenv('DEBUG'):
            for name, param in doc_model.named_parameters():
                if param.grad is None:
                    print(f"No grad for {name}")
                else:
                    print(f"Grad for {name}: {param.grad.norm()}")

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f">>> Epoch {epoch+1} average loss: {avg_loss:.4f}")
    # optional: wandb.log({"train_loss": avg_loss, "epoch": epoch})

torch.save(query_model.state_dict(), 'data/query_model.pt')
save_model('query_model', 'The trained model for our queries')

torch.save(doc_model.state_dict(), 'data/doc_model.pt')
save_model('doc_model', 'The trained model for our documents')
