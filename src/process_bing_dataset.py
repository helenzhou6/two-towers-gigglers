import json
from datasets import load_dataset
import pandas as pd
import os
from fasttext.FastText import tokenize
from utils import init_wandb, load_model_path, save_model

init_wandb()

# Ensure "data" folder exists
os.makedirs("data", exist_ok=True)

ds = load_dataset("microsoft/ms_marco", "v1.1")
training_df = ds["train"].to_pandas()

vocab_path = load_model_path('vocab:latest')
with open(vocab_path) as file:
    word2idx = json.load(file)

passage_text_series = training_df["passages"].apply(lambda row: row["passage_text"])


# -- DOC DATASET
# Unique rows, each row is a doc (string)
all_doc_list = training_df["passages"].apply(lambda row: row["passage_text"]).explode().drop_duplicates().apply(lambda text: [word2idx.get(tok, word2idx.get("<UNK>")) for tok in tokenize(text)])
doc_data = {
    'doc': all_doc_list,
}
docs_df = pd.DataFrame(doc_data)
docs_df.to_csv('data/query_processed.csv', index=False)
save_model('query_processed', 'The processed query word indexes', 'csv', type='dataset')
print('processed query...')

# -- QUERY DATASET
query_data = {
    "query": training_df["query"],
    "doc": passage_text_series
}

def tokenize_row(row):
    row['query'] = [word2idx.get(word, word2idx.get("<UNK>")) for word in tokenize(row['query'])]
    row['doc'] = [word2idx.get(word, word2idx.get("<UNK>")) for word in tokenize(row['doc'])]
    return row

query_df = pd.DataFrame(query_data).explode("doc").apply(tokenize_row, axis=1)
query_df.to_csv('data/docs_processed.csv', index=False)
save_model('docs_processed', 'The processed document word indexes', 'csv', type='dataset')
print('processed docs...')
