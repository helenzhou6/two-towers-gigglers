import torch

class QryTower(torch.nn.Module):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(10, 1)
        self.embedding = embedding

    def forward(self, x):
        # TODO: use embedding
        x = self.fc(x)
        return x
    
class DocTower(torch.nn.Module):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(10, 1)
        self.embedding = embedding

    def forward(self, x):
        # TODO: use embedding
        x = self.fc(x)
        return x