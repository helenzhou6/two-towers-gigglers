import torch
import fasttext
from collections import Counter
import pandas as pd
from fasttext.FastText import tokenize

# === 0. Params ===
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM =  300
QUERY_DATA_FILE_PATH = 'data/query.parquet'
DOCS_DATA_FILE_PATH = 'data/docs.parquet'

fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

# === 1. Sample corpus ===
# TODO: SWAP OUT THE BELOW!
# corpus = pd.concat([pd.read_parquet(query_data_file)["query"].drop_duplicates(), pd.read_parquet(docs_data_file)["doc"]])
corpus = pd.concat([pd.read_parquet(QUERY_DATA_FILE_PATH)["query"].drop_duplicates()[:10], pd.read_parquet(DOCS_DATA_FILE_PATH)["doc"][:10]])

# # === 2. Tokenize ===
corpus_tokenized = corpus.apply(lambda row: tokenize(row)).sum()

# # === 3. Count words and build vocabulary ===
counter = Counter(corpus_tokenized)
most_common = counter.most_common(MAX_VOCAB_SIZE)
word2idx = {word: idx for idx, (word, _) in enumerate(most_common)}
word2idx['<UNK>'] = len(word2idx)  # Add unknown token

vocab_size = len(word2idx)
print("Vocab size:", vocab_size)

# # === 4. Load FastText model ===
model = fasttext.load_model('cc.en.300.bin')  # Assumes you downloaded this

# # === 5. Create fasttext_tensor ===
vectors = []
for word in word2idx:
    vec = model.get_word_vector(word)
    vectors.append(torch.tensor(vec))
fasttext_tensor = torch.stack(vectors)  # shape: [vocab_size, EMBEDDING_DIM]
print(fasttext_tensor.shape)

# # === 6. Create Embedding layer ===
# embedding_layer = torch.nn.Embedding.from_pretrained(fasttext_tensor, freeze=False)

# # === 7. Example usage ===
# sample_sentence = ["the", "cat", "sat", "on", "rug"]  # Note: "rug" is OOV
# indices = [word2idx.get(word, word2idx['<unk>']) for word in sample_sentence]
# input_tensor = torch.tensor(indices)

# embedded = embedding_layer(input_tensor)
# print("Embedded shape:", embedded.shape)  # [len(sentence), 300]