import numpy as np
from utils import get_device
from torch.utils.data import Dataset
import torch

device = get_device()

class KeyQueryDataset(Dataset):
    def __init__(self, start, end, word2idx, queries, positives, negatives):
        super().__init__()
        self.UNK = word2idx["<UNK>"]
        # Store data as tensors up front to avoid per-batch conversion overhead
        tensorize = lambda seqs: [torch.tensor(s, dtype=torch.long) for s in seqs]

        self.queries = tensorize(queries)
        self.positives = tensorize(positives)
        self.negatives = tensorize(negatives)

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
