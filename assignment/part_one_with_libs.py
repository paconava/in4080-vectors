from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from nltk import ngrams


def get_count_vectorizer(documents: List[str]) -> CountVectorizer:
    """
    Method that returns a fitted sklearn.feature_extraction.text.CountVectorizer.
    We will use the CountVectorizer for getting term frequencies.

    Important: The default settings for CountVecvtorizer is to only consider
    terms with more than one characters and only alphanumericals as words.
    We want to count all types of tokens. You can accomplish this passing
    token_pattern='\S+' as argument to the constructor.

    :param document: list of strings (words)
    :return: CountVectorizer, fitted to the documents
    """
    return CountVectorizer(token_pattern='\S+').fit(documents)

def get_tfidf_transformer(documents: List[str], idf=None) -> TfidfTransformer:
    """
    Method that, for each document in documents, returns a term frequency (word count) dict.
    The return value should thus be a list of dictionaries, with strings as keys and integers as values.

    Note: We could have used TfidfVectorizer here, which is a combination of CountVectorizer + TfidfTransformer.
    But since we want to mirror what we did in the previous part, we combine the two objects explicitly in the exercise.

    :param documents: list of documents, where each document is a list of words
    :return: TfidfTransformer, fitted to the documents
    """
    return TfidfTransformer(smooth_idf=False, norm=None).fit(documents)

# def cosine_similarity(x, y):
#     """
#     We'll just reuse sklearn's cosine_similarity_function.

#     For cleaner code, you may actually delete this method and import the method instead:
#     from sklearn.metrics.pairwise import cosine_similarity

#     :param x:
#     :param y:
#     :return:
#     """
#     return pairwise.cosine_similarity(x, y)


def get_most_similar_pair(tfidfs: np.ndarray) -> Tuple[float, int, int]:
    """
    Get the most similar TFIDF vectors, using Cosine as similarity metric.

    Hint: Use sklearn.metrics.pairwise.cosine_similarity

    :param tfidfs: TFIDF scores for a list of dicuments
    :return: a tuple with (score, index_of_first_vector, index_of_second_vector
    """
    vec1index = 0
    vec2index = 0
    winningscore = -1
    n, m = tfidfs.shape
    for i in range(0,n-1):
        for j in range(i+1,n):
            score = cosine_similarity(tfidfs[i], tfidfs[j])[0][0]
            if score > winningscore:
                winningscore = score
                vec1index = i
                vec2index = j
    return (winningscore, vec1index, vec2index)

def reverse_word2index(word2index: Dict[str, int]) -> List[str]:
    """
    In order to map from indexes to words, we need to reverse the word2index mapping.
    This time, we'll do with a list of strings.
    Reminder: An index is a pointer to a position in a list/vector.

    Hint: You can reuse your solution from the first part

    :param word2index:
    :return: list of words
    """
    a = [0 for i in word2index.items()]
    for k, v in word2index.items():
        a[v] = k
    return a


def extract_contexts(documents: List[List[str]], word2index: Dict[str, int], window_size: int) -> np.ndarray:
    """
    Create a matrix (a list of lists here) where:
    - the indices in the outer dimension are the indices of the target words
    - the indices in the inner dimension are the indices of the context words
    - the window_size parameter defines how many words we count before and after the target word
      Example: Window size 2 should get 4 context words for a target word, 2 before and 2 after.
    - we don't do any padding, so for the first word of a sentence, we will only count the words after the target word.

    Hint: Use nltk.ngrams in this method

    :param documents: documents at sentence level
    :param word2index: word2index dict, in order to map the words to the indices
    :param window_size: defines the size of the context, so we take *size* words before and *size* words after the target word.
    :return: context term frequencies as a matrix of ints
    """

    # I'll help you initialize the context matrix. You're welcome ;)
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


def get_most_similar_to(idx: int, tfidfs: np.ndarray) -> Tuple[float, int]:
    """
    Get the most similar TFIDF vectors, using Cosine as similarity metric.
    Reminder: An index is a pointer to a position in a list/vector.

    Hint: Use sklearn.metrics.pairwise.cosine_similarity

    :param idx: index of the word we want to compare similarities to
    :param tfidfs: TF-IDF scores as a numpy array/matrix
    :return: a tuple with (score, index_of_first_vector, index_of_second_vector
    """
    n,m = tfidfs.shape
    vec2index = 0
    winningscore = -1
    for i in range(0,n):
        if i != idx:
            score = cosine_similarity(tfidfs[i], tfidfs[idx])[0][0]
            if score > winningscore:
                winningscore = score
                vec2index = i
    return (winningscore, vec2index)
