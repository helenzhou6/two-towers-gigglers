import random
from utils import load_artifact_path, load_model_path
from torch.utils.data import IterableDataset, DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from fasttext.FastText import tokenize

class KeyQueryDataset(IterableDataset):
    def __init__(self, start, end, word2idx):
        super().__init__()
        self.UNK = word2idx["<UNK>"]

        # Pre-tokenize and convert to index lists once
        queries_path = load_artifact_path('query_processed', file_extension='parquet')
        queries_processed = pd.read_parquet(queries_path)
        
        documents_path = load_artifact_path('docs_processed', file_extension='parquet')
        documents_processed = pd.read_parquet(documents_path)

        print(queries_processed.columns)

        self.queries = queries_processed['query']
        self.positives = queries_processed["doc"]
        self.negatives = documents_processed["doc"]

        self.nq = len(self.queries)
        self.nd = len(self.negatives)
        self.start, self.end = start, end

    def __iter__(self):
        for _ in range(self.start, self.end):
            i = random.randrange(self.nq)
            j = random.randrange(self.nd)
            yield (
                torch.tensor(self.queries[i]),
                torch.tensor(self.positives[i]),
                torch.tensor(self.negatives[j]),
            )

    def __len__(self):
        return self.end - self.start

def collate_fn_emb_bag(data_items):
    if not data_items:
        return None
    
    queries, pos_samples, neg_samples = zip(*data_items)
    
    cat_queries = torch.cat(queries)
    cat_pos_samples = torch.cat(pos_samples)
    cat_neg_samples = torch.cat(neg_samples)

    query_lengths = torch.tensor([t.numel() for t in queries], dtype=torch.long)
    # The offsets are the cumulative sum of the lengths, starting with 0
    query_offsets = torch.cat([torch.tensor([0]), query_lengths.cumsum(dim=0)[:-1]])

    pos_samples_lengths = torch.tensor([t.numel() for t in pos_samples], dtype=torch.long)
    pos_samples_offsets = torch.cat([torch.tensor([0]), pos_samples_lengths.cumsum(dim=0)[:-1]])
    
    neg_samples_lengths = torch.tensor([t.numel() for t in neg_samples], dtype=torch.long)
    neg_samples_offsets = torch.cat([torch.tensor([0]), neg_samples_lengths.cumsum(dim=0)[:-1]])

    return ((cat_queries, query_offsets), (cat_pos_samples, pos_samples_offsets), (cat_neg_samples, neg_samples_offsets))

def getKQDataLoader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn_emb_bag)