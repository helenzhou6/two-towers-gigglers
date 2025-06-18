import random
import datetime
from utils import get_device, load_artifact_path
from torch.utils.data import IterableDataset, DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from fasttext.FastText import tokenize

device = get_device()

class KeyQueryDataset(IterableDataset):
    def __init__(self, start, end, word2idx):
        super().__init__()
        self.UNK = word2idx["<UNK>"]

        # Pre-tokenize and convert to index lists once
        queries_path = load_artifact_path('query_processed', file_extension='parquet')
        queries_processed = pd.read_parquet(queries_path)
        
        documents_path = load_artifact_path('docs_processed', file_extension='parquet')
        documents_processed = pd.read_parquet(documents_path)

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
                self.queries[i],
                self.positives[i],
                self.negatives[j],
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

import itertools

def _flatten_and_offsets(seqs):
    flat = list(itertools.chain.from_iterable(seqs))
    lengths = [len(s) for s in seqs]
    offsets = [0]
    for l in lengths[:-1]:
        offsets.append(offsets[-1] + l)
    return flat, offsets

def collate_fn_emb_bag_py(data_items):
    if not data_items:
        return None

    # unzip into three tuples of lists
    queries, pos_samples, neg_samples = zip(*data_items)

    # helper to flatten + offsets

    flat_q, q_off = _flatten_and_offsets(queries)
    flat_p, p_off = _flatten_and_offsets(pos_samples)
    flat_n, n_off = _flatten_and_offsets(neg_samples)

    return (flat_q, q_off), (flat_p, p_off), (flat_n, n_off)
