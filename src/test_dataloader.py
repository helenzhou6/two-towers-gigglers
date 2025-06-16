from dataloader import KeyQueryDataset, getKQDataLoader
import json

W2IX_FILE = 'data/vocab.json'

w2ix = json.load(W2IX_FILE)
dataset = KeyQueryDataset(w2ix)
print(next(dataset))

dataloader = getKQDataLoader(dataset, 3)

print(next(dataloader))