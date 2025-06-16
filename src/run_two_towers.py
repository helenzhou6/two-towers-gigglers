import torch
from two_towers import QryTower, DocTower

# Init Two Towers
qryTower = QryTower()
docTower = DocTower()

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

