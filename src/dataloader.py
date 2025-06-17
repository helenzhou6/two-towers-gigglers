import random
from torch.utils.data import IterableDataset, DataLoader
import torch
import pandas as pd
from fasttext.FastText import tokenize

class KeyQueryDataset(IterableDataset):
    def __init__(self, start, end, word2idx, query_data_file, docs_data_file):
        super().__init__()
        self.UNK = word2idx["<UNK>"]
        df_q = pd.read_parquet(query_data_file)
        df_d = pd.read_parquet(docs_data_file)

        # Pre-tokenize and convert to index lists once
        self.queries = [
            torch.tensor([word2idx.get(tok, self.UNK) for tok in tokenize(q)])
            for q in df_q["query"]
        ]
        self.positives = [
            torch.tensor([word2idx.get(tok, self.UNK) for tok in d])
            for d in df_q["doc"]
        ]
        self.documents = [
            torch.tensor([word2idx.get(tok, self.UNK) for tok in d])
            for d in df_d["doc"]
        ]

        self.nq = len(self.queries)
        self.nd = len(self.documents)
        self.start, self.end = start, end

    def __iter__(self):
        for _ in range(self.start, self.end):
            i = random.randrange(self.nq)
            j = random.randrange(self.nd)
            yield (
                self.queries[i],
                self.positives[i],
                self.documents[j],
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