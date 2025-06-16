from torch.utils.data import IterableDataset, DataLoader
import torch
import pandas as pd
from fasttext.FastText import tokenize

class KeyQueryDataset(IterableDataset):
    def __init__(self,   word2idx =None,  query_data_file= 'data/query.parquet', 
                 docs_data_file = 'data/docs.parquet', ): #num_negative_samples = 1,
        super().__init__()
        #TODO: Allow num_negative_samples to be set bigger than one
        self.word2idx = word2idx
        #self.num_neg_samples = num_negative_samples
        self.query_data = pd.read_parquet(query_data_file)
        self.doc_data = pd.read_parquet(docs_data_file)
    
    def __iter__(self):
        #TODO: Set up with multiple workers
        #get positive sample
        query, pos_sample = tuple(self.query_data.sample(1).loc[:,["query", "doc"]])
        #get negative samples
        neg_sample = self.doc_data.sample(1)["doc"][0]
        query_indices = torch.tensor([self.word2idx.get(token, '<UNK>') for token in tokenize(query)])
        pos_sample_indices = torch.tensor([self.word2idx.get(token, '<UNK>') for token in tokenize(pos_sample)])
        neg_sample_indices = torch.tensor([self.word2idx.get(token, '<UNK>') for token in tokenize(neg_sample)])

        yield (query_indices, pos_sample_indices, neg_sample_indices)

def collate_fn_emb_bag(data_items):
    queries, pos_samples, neg_samples = zip(data_items)
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