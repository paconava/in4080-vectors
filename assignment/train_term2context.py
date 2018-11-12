"""
Train a Term2Context TFIDF model, and print the most similar words to "war"
"""

# Train a model (remember you can reuse code here)
import warnings
import operator
warnings.filterwarnings("ignore")
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

import nltk
from nltk.corpus import inaugural

def get_inaugural_docs(download=False) -> List[List[List[str]]]:
    """
    Get the inaugural documents as a list (documents) of list (sentences) of list (sentence) of strings (words)
    :param download: If True, the corpus will be downloaded. Default=False
    :return:
    """
    if download:
        nltk.download('inaugural')
    return [[[w.lower() for w in sent] for sent in inaugural.sents(fileid)] for fileid in inaugural.fileids()]

def flatten_sentences(document: List[List[str]]) -> List[str]:
    """
    Method to take document as a list of sentences of words, and flatten it into a list of words
    :param document: list of list of strings (words)
    :return: list of strings
    """
    return [w for sent in document for w in sent]

def get_count_vectorizer(documents: List[str]) -> CountVectorizer:
    """
    Method that returns a fitted sklearn.feature_extraction.text.CountVectorizer.
    :param document: list of strings (words)
    :return: CountVectorizer, fitted to the documents
    """
    return CountVectorizer(token_pattern='\S+').fit(documents)

def get_tfidf_transformer(documents: List[str], idf=None) -> TfidfTransformer:
    """
    Method that, for each document in documents, returns a term frequency (word count) dict.
    :param documents: list of documents, where each document is a list of words
    :return: TfidfTransformer, fitted to the documents
    """
    return TfidfTransformer(smooth_idf=False, norm=None).fit(documents)

def extract_contexts(documents: List[List[str]], word2index: Dict[str, int], window_size: int) -> np.ndarray:
    """
    Create a matrix (a list of lists here) where:
    - the indices in the outer dimension are the indices of the target words
    - the indices in the inner dimension are the indices of the context words
    - the window_size parameter defines how many words we count before and after the target word
      Example: Window size 2 should get 4 context words for a target word, 2 before and 2 after.
    - we don't do any padding, so for the first word of a sentence, we will only count the words after the target word.
    :param documents: documents at sentence level
    :param word2index: word2index dict, in order to map the words to the indices
    :param window_size: defines the size of the context, so we take *size* words before and *size* words after the target word.
    :return: context term frequencies as a matrix of ints
    """
    matrix = np.zeros((len(word2index), len(word2index)))
    for i in documents:
        for j in i:
            for w in range(0,len(j)):
                for mov in range(1,window_size+1):
                    if (w+mov) <= (len(j)-1):
                        matrix[word2index[j[w]]][word2index[j[w+mov]]] += 1
                    if (w-mov) >= 0:
                        matrix[word2index[j[w]]][word2index[j[w-mov]]] += 1
    return matrix

def reverse_word2index(word2index: Dict[str, int]) -> List[str]:
    """
    In order to map from indexes to words, we need to reverse the word2index mapping.
    :param word2index:
    :return: list of words
    """
    a = [0 for i in word2index.items()]
    for k, v in word2index.items():
        a[v] = k
    return a

def get_most_similar_to(idx: int, tfidfs: np.ndarray) -> Tuple[float, int]:
    """
    Get the most similar TFIDF vectors, using Cosine as similarity metric.
    :param idx: index of the word we want to compare similarities to
    :param tfidfs: TF-IDF scores as a numpy array/matrix
    :return: a tuple with (score, index_of_first_vector, index_of_second_vector
    """
    n,m = tfidfs.shape
    vec2index = [0 for i in range(0,5)]
    winningscore = [-1 for i in range(0,5)]
    minws = 0
    minidx = 0
    for i in range(0,n):
        if i != idx:
            score = cosine_similarity(tfidfs[i], tfidfs[idx])[0][0]
            minws = min(winningscore)
            minidx = winningscore.index(minws)
            if score > minws:
                winningscore[minidx] = score
                vec2index[minidx] = i
    return (winningscore, vec2index)

documents_as_sentences = get_inaugural_docs()
documents_as_words = [flatten_sentences(doc) for doc in documents_as_sentences]
documents_as_raw_text = [' '.join(doc) for doc in documents_as_words]

cv = get_count_vectorizer(documents_as_raw_text)
tfs = cv.transform(documents_as_raw_text)

term2index = cv.vocabulary_

contexts = extract_contexts(documents_as_sentences, term2index, 1)
index2term = reverse_word2index(term2index)
tfidf_transformer = get_tfidf_transformer(contexts)
tfidfs = tfidf_transformer.transform(contexts)
best_similarity, most_similar_idx = get_most_similar_to(term2index['war'], tfidfs)
most_similar_to_war = {}
for i in range(0,len(most_similar_idx)):
    most_similar_to_war[index2term[most_similar_idx[i]]] = best_similarity[i]
print("5 most similar words to war:")
fmtprnt = sorted(most_similar_to_war.items(), key=operator.itemgetter(1), reverse=True)
print(fmtprnt)

# caption = '5 most similar words to war'
# label = 'tab:t2c'
# tex_format.print_table(3,5,['', 'Word', 'Score'],fmtprnt, caption, label)