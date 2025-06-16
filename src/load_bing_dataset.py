from datasets import load_dataset
import pandas as pd
ds = load_dataset("microsoft/ms_marco", "v1.1")
training_df = ds["train"].to_pandas()

# OUTPUT: Doc dataset pandas
# ALL document names - i.e. all the passage_text, extract and remove duplicates
all_doc_list = training_df["passages"].apply(lambda row: row["passage_text"]).explode()
data = {
    'Doc': all_doc_list,
}
docs_df = pd.DataFrame(data).drop_duplicates()

# Query dataset: For each row/query - query (string), doc_title (string), is_clicked = true