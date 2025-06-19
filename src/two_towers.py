import torch

class QryTower(torch.nn.Module):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_dim = embedding.embedding_dim
        self.embedding = embedding
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )
        
    def forward(self, x):
        return self.layers(self.embedding(*x))

class DocTower(torch.nn.Module):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_dim = embedding.embedding_dim
        self.embedding = embedding
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )

    def forward(self, x):
        return self.layers(self.embedding(*x))