import torch
import fasttext
from collections import Counter
import pandas as pd
from fasttext.FastText import tokenize
from fasttext import util
import json
from tqdm import tqdm

from utils import init_wandb, save_model

tqdm.pandas()

# === 0. Params ===
MAX_VOCAB_SIZE = 100000
EMBEDDING_DIM =  300
QUERY_DATA_FILE_PATH = 'data/query.parquet'
DOCS_DATA_FILE_PATH = 'data/docs.parquet'

init_wandb(None, None)

util.download_model('en', if_exists='ignore')  # English
fasttext_model = fasttext.load_model('cc.en.300.bin')

# === 1. Sample corpus ===
print('START: Concat Corpuses')
corpus = pd.concat([pd.read_parquet(QUERY_DATA_FILE_PATH)["query"].drop_duplicates(), pd.read_parquet(DOCS_DATA_FILE_PATH)["doc"]])
print('DONE: Concat Corpuses')

# # === 2. Tokenize ===

print('START: Tokenize Corpuses')
corpus_tokenized = corpus.progress_apply(lambda row: tokenize(row)).explode().value_counts()
print('DONE: Tokenize Corpuses')

# # === 3. Count words and build vocabulary ===
print('START: Count words and build most common vocab')
most_common = corpus_tokenized.head(MAX_VOCAB_SIZE)

word2idx = {}
ind = 0
for word in most_common.index.tolist():
    word2idx[word] = ind
    ind += 1
word2idx['<UNK>'] = len(word2idx)  # Add unknown token
print('END: Count words and build most common vocab')

print(f"START: writing vocab.json for {MAX_VOCAB_SIZE} words")
with open("data/vocab.json", "w") as file:
    json.dump(word2idx, file)

vocab_size = len(word2idx)
print("Vocab size:", vocab_size)
print(f"END: writing vocab.json for {MAX_VOCAB_SIZE} words")

# # === 5. Create fasttext_tensor ===
print(f"START: create vectors for tokenized words")
vectors = []
for word in tqdm(word2idx):
    vec = fasttext_model.get_word_vector(word)
    vectors.append(torch.tensor(vec))
fasttext_tensor = torch.stack(vectors)  # shape: [vocab_size, EMBEDDING_DIM]
print(f"DONE: create vectors for tokenized words")

# # === 6. Create Embedding layer ===
embedding_layer = torch.nn.Embedding.from_pretrained(fasttext_tensor, freeze=False)
torch.save(embedding_layer.state_dict(), 'data/fasttext_tensor.pt')
save_model('fasttext_tensor', 'FastText embedding from data set')
save_model('vocab', 'The indicies of our words in the vocabulary', file_extension='json')