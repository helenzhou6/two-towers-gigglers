import json
from datasets import load_dataset
import pandas as pd
import os
from fasttext.FastText import tokenize
from utils import init_wandb, load_model_path, save_artifact
import argparse

init_wandb()

def process_and_save_dataset(SPLIT):
    # Ensure "data" folder exists
    os.makedirs("data", exist_ok=True)

    ds = load_dataset("microsoft/ms_marco", "v1.1")
    split_df = ds[SPLIT].to_pandas()

    vocab_path = load_model_path('vocab:latest')
    with open(vocab_path) as file:
        word2idx = json.load(file)

    passage_text_series = split_df["passages"].apply(lambda row: row["passage_text"])

    # -- DOC DATASET
    # Unique rows, each row is a doc (string)
    all_doc_list = split_df["passages"].apply(lambda row: row["passage_text"]).explode().drop_duplicates().apply(lambda text: [word2idx.get(tok, word2idx.get("<UNK>")) for tok in tokenize(text)])
    doc_data = {
        'doc': all_doc_list,
    }
    docs_df = pd.DataFrame(doc_data)
    if SPLIT == 'train':
        docs_file_name = f'docs_processed'
    else:
        docs_file_name = f'docs_processed_{SPLIT}'

    docs_df.to_parquet(f"data/{docs_file_name}.parquet", index=False)
    save_artifact(docs_file_name, 'The processed docs word indexes', 'parquet', type='dataset')
    print('processed docs...')

    # -- QUERY DATASET
    query_data = {
        "query": split_df["query"],
        "doc": passage_text_series
    }

    def tokenize_row(row):
        row['query'] = [word2idx.get(word, word2idx.get("<UNK>")) for word in tokenize(row['query'])]
        row['doc'] = [word2idx.get(word, word2idx.get("<UNK>")) for word in tokenize(row['doc'])]
        return row

    if SPLIT == 'train':
        query_file_name = f'query_processed'
    else:
        query_file_name = f'query_processed_{SPLIT}'

    query_df = pd.DataFrame(query_data).explode("doc").apply(tokenize_row, axis=1)
    query_df.to_parquet(f"data/{query_file_name}.parquet", index=False)
    save_artifact(query_file_name, 'The processed query word indexes', 'parquet', type='dataset')
    print('processed query...')

def main():
    parser = argparse.ArgumentParser(description="Process MS MARCO dataset.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (e.g., 'train', 'test', 'validation')",
    )
    args = parser.parse_args()
    process_and_save_dataset(args.split)

if __name__ == "__main__":
    main()
