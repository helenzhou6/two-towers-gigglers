from datasets import load_dataset
import pandas as pd
import os

# Ensure "data" folder exists
os.makedirs("data", exist_ok=True)

ds = load_dataset("microsoft/ms_marco", "v1.1")
training_df = ds["train"].to_pandas()

# -- DOC DATASET
# Unique rows, each row is a doc (string)
passage_text_series = training_df["passages"].apply(lambda row: row["passage_text"])
all_doc_list = training_df["passages"].apply(lambda row: row["passage_text"]).explode()
doc_data = {
    'doc': all_doc_list,
}
docs_df = pd.DataFrame(doc_data).drop_duplicates()
docs_df.to_parquet("data/docs.parquet", index=False)

# -- QUERY DATASET
# Each row has: query (string), doc (string), clicked (0/1)
query_data = {
    "query": training_df["query"],
    "clicked": training_df["passages"].apply(lambda row: row["is_selected"]),
    "doc": passage_text_series
}

query_df = pd.DataFrame(query_data).explode("doc").explode("clicked").reset_index(drop=True)
query_df.to_parquet("data/query.parquet", index=False)