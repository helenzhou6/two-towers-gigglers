import torch

class QryTower(torch.nn.Module):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding
        self.fc = torch.nn.Linear(self.embedding.embedding_dim, 1)

    def forward(self, x_indices, x_offsets):
        # x_indices: concatenated token indices
        # x_offsets: where each sentence starts
        x = self.embedding(x_indices, x_offsets)  # shape: (batch_size, embedding_dim)
        x = self.fc(x)                            # shape: (batch_size, 1)
        return x
    
class DocTower(torch.nn.Module):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(self.embedding.embedding_dim, 1)
        self.embedding = embedding

    def forward(self, x_indices, x_offsets):
        # x_indices: concatenated token indices
        # x_offsets: where each sentence starts
        x = self.embedding(x_indices, x_offsets)  # shape: (batch_size, embedding_dim)
        x = self.fc(x)                            # shape: (batch_size, 1)
        return x