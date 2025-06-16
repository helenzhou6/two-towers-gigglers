import fasttext
import fasttext.util
import torch
import re

fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def sentence_to_tensor(text, model):
    tokens = fasttext.tokenize(text)
    vectors = [torch.tensor(model.get_word_vector(token)) for token in tokens]
    return vectors

def average_vector(vectors):
    if not vectors:
        return torch.zeros(300)
    return torch.stack(vectors).mean(dim=0)


print(sentence_to_tensor("what is the vector space", ft))