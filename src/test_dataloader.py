from dataloader import KeyQueryDataset, getKQDataLoader
import json

# May need to change the 
W2IX_FILE_PATH = 'artifacts/vocab:v0/vocab.json'
with open(W2IX_FILE_PATH) as file:
    w2ix = json.load(file)
dataset = KeyQueryDataset(0, 2, w2ix)
datasample = next(iter(dataset))
# print(datasample)
# print(type(datasample), len(datasample))
dataloader = getKQDataLoader(dataset, batch_size=3)

# TODO: Feed into EmbeddingBag
print(next(iter(dataloader)))
