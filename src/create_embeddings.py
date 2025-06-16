import torch
import fasttext
from collections import Counter
import re

# === 1. Sample corpus ===
corpus = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "The mat was sat on by the dog."
]

# === 2. Tokenize ===
tokenized = [re.findall(r'\b\w+\b', sentence.lower()) for sentence in corpus]
flat_tokens = [word for sentence in tokenized for word in sentence]

# === 3. Count words and build vocabulary ===
counter = Counter(flat_tokens)
MAX_VOCAB_SIZE = 10000
most_common = counter.most_common(MAX_VOCAB_SIZE)

word2idx = {word: idx for idx, (word, _) in enumerate(most_common)}
word2idx['<unk>'] = len(word2idx)  # Add unknown token

vocab_size = len(word2idx)
print("Vocab size:", vocab_size)

# === 4. Load FastText model ===
model = fasttext.load_model('cc.en.300.bin')  # Assumes you downloaded this

# === 5. Create fasttext_tensor ===
embedding_dim = 300
vectors = []
for word in word2idx:
    vec = model.get_word_vector(word)
    vectors.append(torch.tensor(vec))

fasttext_tensor = torch.stack(vectors)  # shape: [vocab_size, embedding_dim]

# === 6. Create Embedding layer ===
embedding_layer = torch.nn.Embedding.from_pretrained(fasttext_tensor, freeze=False)

# === 7. Example usage ===
sample_sentence = ["the", "cat", "sat", "on", "rug"]  # Note: "rug" is OOV
indices = [word2idx.get(word, word2idx['<unk>']) for word in sample_sentence]
input_tensor = torch.tensor(indices)

embedded = embedding_layer(input_tensor)
print("Embedded shape:", embedded.shape)  # [len(sentence), 300]