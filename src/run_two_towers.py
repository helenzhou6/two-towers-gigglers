import torch.nn.functional as F
import torch
import json
from two_towers import QryTower, DocTower
from utils import load_model_path, init_wandb

EPOCHS = 5

# Loads embeddings and test out works
init_wandb(5)
ft_embedded_path = load_model_path('fasttext_tensor:latest')
ft_state_dict = torch.load(ft_embedded_path)
num_embeddings, embedding_dim = ft_state_dict["weight"].shape
embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean")
embedding_bag.load_state_dict(ft_state_dict) # Returns Embedding(100001, 300)

# Init Two Towers
qryTower = QryTower(embedding_bag)
docTower = DocTower(embedding_bag)

# Given Torch vector (1, 10)
qry = torch.randn(1, 10)
pos = torch.randn(1, 10)
neg = torch.randn(1, 10)

# Run the model and output Wd and Wd(pos) + Wd(neg)
qry = qryTower(qry)
pos = docTower(pos)
neg = docTower(neg)

# Creates positive and negative score 
dst_pos_score = torch.nn.functional.cosine_similarity(qry, pos)
dst_neg_score = torch.nn.functional.cosine_similarity(qry, neg)
# Difference between the two scores
dst_dif = dst_pos_score - dst_neg_score
# Margin
dst_mrg = torch.tensor(0.2)

# Loss function
# Makes the negative doc within the margin
loss = torch.max(torch.tensor(0.0), dst_mrg - dst_dif)
# To create the gradients for training
loss.backward()

# TODOs:
# -- Need the dataloader to function, to be able input the raw data into both models
# -- Link the two towers to output a similarity score
# -- Train it