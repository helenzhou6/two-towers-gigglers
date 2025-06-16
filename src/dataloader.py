from torch.utils.data import IterableDataset, DataLoader
import torch
import pandas as pd
from fasttext.FastText import tokenize

class KeyQueryDataset(IterableDataset):

    def __init__(self,   word2idx =None, num_negative_samples = 1, query_data_file= 'data/query.parquet', 
                 docs_data_file = 'data/docs.parquet', ):
        super().__init__()
    
        self.word2idx = word2idx
        self.num_neg_samples = num_negative_samples
        self.query_data = pd.load_parquet(query_data_file)
        self.doc_data = pd.load_parquet(docs_data_file)
    
    def __iter__(self):
        #TODO: Set up with multiple workers
        #get positive sample
        query, pos_sample = tuple(self.query_data.sample(1).loc[:,["query", "doc"]])
        #get negative samples
        neg_samples = list(self.doc_data.sample(self.num_neg_samples)["doc"])
        query_indices = torch.tensor([self.word2idx.get(token, '<UNK>') for token in tokenize(query)])
        pos_sample_indices = torch.tensor([self.word2idx.get(token, '<UNK>') for token in tokenize(pos_sample)])
        neg_samples_indices = [torch.tensor([self.word2idx.get(token, '<UNK>') for token in tokenize(neg_sample)]
                       for neg_sample in neg_samples)]
        #TODO: Transform to index before returning
        yield (query_indices, pos_sample_indices, neg_samples_indices)
