from dataloader import KeyQueryDataset, getKQDataLoader
import json

W2IX_FILE = 'data/vocab.json'
with open(W2IX_FILE) as file:
    w2ix = json.load(file)
dataset = KeyQueryDataset(w2ix)
datasample = next(iter(dataset))
print(datasample)
print(type(datasample), len(datasample))
dataloader = getKQDataLoader(dataset, batch_size=3)

print(next(iter(dataloader)))
