import random
from utils import load_artifact_path, load_model_path
from torch.utils.data import IterableDataset, DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from fasttext.FastText import tokenize

class KeyQueryDataset(IterableDataset):
    # def __init__(self, start, end, word2idx=None, query_data_file='data/query.parquet', 
    #              docs_data_file='data/docs.parquet'): #num_negative_samples = 1,
    #     super().__init__()
    #     #TODO: Allow num_negative_samples to be set bigger than one
    #     self.word2idx = word2idx
    #     self.UNK_val = self.word2idx.get("<UNK>")
    #     self.start = start
    #     self.end = end
    #     #self.num_neg_samples = num_negative_samples
    #     self.query_data = pd.read_parquet(query_data_file)
    #     self.doc_data = pd.read_parquet(docs_data_file)
    
    # def __iter__(self):
    #     #TODO: Set up with multiple workers
    #     for _ in range(self.start, self.end):
            
    #         # get positive sample
    #         query, pos_sample = tuple(self.query_data.sample(1).iloc[0][["query", "doc"]])

    #         # get negative samples
    #         neg_sample = self.doc_data.sample(1)["doc"].iloc[0]

    #         query_indices = torch.tensor([self.word2idx.get(token, self.UNK_val) for token in tokenize(query)])
    #         pos_sample_indices = torch.tensor([self.word2idx.get(token, self.UNK_val) for token in tokenize(pos_sample)])
    #         neg_sample_indices = torch.tensor([self.word2idx.get(token, self.UNK_val) for token in tokenize(neg_sample)])

    #         yield (query_indices, pos_sample_indices, neg_sample_indices)

    def __init__(self, start, end, word2idx):
        super().__init__()
        self.UNK = word2idx["<UNK>"]

        # Pre-tokenize and convert to index lists once
        queries_path = load_artifact_path('queries_processed')
        queries_processed = pd.read_csv(queries_path)
        
        documents_path = load_artifact_path('docs_processed')
        documents_processed = pd.read_csv(documents_path)

        self.queries = queries_processed['queries']
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

def getKQDataLoader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn_emb_bag)