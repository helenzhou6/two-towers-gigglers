import numpy as np
import datetime
from utils import get_device, load_artifact_path
from torch.utils.data import IterableDataset, DataLoader, Dataset
import torch
from tqdm import tqdm
import pandas as pd
from fasttext.FastText import tokenize

device = get_device()

class KeyQueryDataset(Dataset):
    def __init__(self, start, end, word2idx):
        super().__init__()
        self.UNK = word2idx["<UNK>"]

        # Pre-tokenize and convert to index lists once
        queries_path = load_artifact_path('query_processed', file_extension='parquet')
        queries_processed = pd.read_parquet(queries_path)
        
        documents_path = load_artifact_path('docs_processed', file_extension='parquet')
        documents_processed = pd.read_parquet(documents_path)

        self.queries = queries_processed['query'].tolist()
        self.positives = queries_processed["doc"].tolist()
        self.negatives = documents_processed["doc"].tolist()

        self.nq = len(self.queries)
        self.nd = len(self.negatives)
        self.start, self.end = start, end
        self.len = end-start

    def __getitem__(self, idx):
        i = np.random.randint(0, self.nq)
        j = np.random.randint(0, self.nd)
        return (
            self.queries[i],
            self.positives[i],
            self.negatives[j],
        )

    def __len__(self):
        return self.len

def collate_fn_emb_bag(data_items):
    """Optimized collate function using torch operations"""
    if not data_items:
        return None
    
    queries, pos_samples, neg_samples = zip(*data_items)
    
    # Convert to tensors if they aren't already
    if not isinstance(queries[0], torch.Tensor):
        queries = [torch.tensor(q, dtype=torch.long) if not isinstance(q, torch.Tensor) else q for q in queries]
        pos_samples = [torch.tensor(p, dtype=torch.long) if not isinstance(p, torch.Tensor) else p for p in pos_samples]
        neg_samples = [torch.tensor(n, dtype=torch.long) if not isinstance(n, torch.Tensor) else n for n in neg_samples]
    
    # Concatenate all sequences
    cat_queries = torch.cat(queries, dim=0)
    cat_pos_samples = torch.cat(pos_samples, dim=0)
    cat_neg_samples = torch.cat(neg_samples, dim=0)

    # Calculate offsets efficiently
    query_lengths = torch.tensor([len(q) for q in queries], dtype=torch.long)
    pos_lengths = torch.tensor([len(p) for p in pos_samples], dtype=torch.long)
    neg_lengths = torch.tensor([len(n) for n in neg_samples], dtype=torch.long)
    
    # Compute offsets using cumsum
    query_offsets = torch.cat([torch.zeros(1, dtype=torch.long), query_lengths.cumsum(0)[:-1]])
    pos_offsets = torch.cat([torch.zeros(1, dtype=torch.long), pos_lengths.cumsum(0)[:-1]])
    neg_offsets = torch.cat([torch.zeros(1, dtype=torch.long), neg_lengths.cumsum(0)[:-1]])

    return ((cat_queries, query_offsets), (cat_pos_samples, pos_offsets), (cat_neg_samples, neg_offsets))

#old collate fn
# def collate_fn_emb_bag(data_items):
#     if not data_items:
#         return None
    
#     queries, pos_samples, neg_samples = zip(*data_items)
    
#     cat_queries = torch.cat(queries)
#     cat_pos_samples = torch.cat(pos_samples)
#     cat_neg_samples = torch.cat(neg_samples)

#     query_lengths = torch.tensor([t.numel() for t in queries], dtype=torch.long)
#     # The offsets are the cumulative sum of the lengths, starting with 0
#     query_offsets = torch.cat([torch.tensor([0]), query_lengths.cumsum(dim=0)[:-1]])

#     pos_samples_lengths = torch.tensor([t.numel() for t in pos_samples], dtype=torch.long)
#     pos_samples_offsets = torch.cat([torch.tensor([0]), pos_samples_lengths.cumsum(dim=0)[:-1]])
    
#     neg_samples_lengths = torch.tensor([t.numel() for t in neg_samples], dtype=torch.long)
#     neg_samples_offsets = torch.cat([torch.tensor([0]), neg_samples_lengths.cumsum(dim=0)[:-1]])


#     return ((cat_queries, query_offsets), (cat_pos_samples, pos_samples_offsets), (cat_neg_samples, neg_samples_offsets))

import itertools

def _flatten_and_offsets(seqs):
    flat = list(itertools.chain.from_iterable(seqs))
    lengths = [len(s) for s in seqs]
    offsets = [0]
    for l in lengths[:-1]:
        offsets.append(offsets[-1] + l)
    return flat, offsets

# def collate_fn_emb_bag_py(data_items):
#     if not data_items:
#         return None

#     # unzip into three tuples of lists
#     queries, pos_samples, neg_samples = zip(*data_items)

#     # helper to flatten + offsets

#     flat_q, q_off = _flatten_and_offsets(queries)
#     flat_p, p_off = _flatten_and_offsets(pos_samples)
#     flat_n, n_off = _flatten_and_offsets(neg_samples)

#     return (flat_q, q_off), (flat_p, p_off), (flat_n, n_off)
