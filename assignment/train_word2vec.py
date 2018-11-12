"""
Train a Word2Vec model, and print the most similar words to "war"
"""
import warnings
warnings.filterwarnings("ignore")
from typing import List, Dict, Tuple
import hashlib
import nltk
from nltk.corpus import inaugural
from gensim.models import Word2Vec

# Pass this hashfxn to gensim's Word2Vec.
def hashfxn(s) -> int:
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8

sentences = inaugural.sents()
model = Word2Vec(sentences=sentences, size=100, workers=1, hashfxn=hashfxn, iter=10)

print("5 most similar words to war:")
print(model.most_similar("war", topn=5))
# Print some more
