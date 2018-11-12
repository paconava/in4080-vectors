from collections import defaultdict
from collections import Counter
from itertools import chain
from itertools import combinations
from math import log, sqrt
from typing import List, Dict, Tuple


def calculate_tf(document: List[str]) -> Dict[str, int]:
    """
    Method that, for each document in documents, returns a term frequency (word count) dict.
    The return value should thus be a list of dictionaries, with strings as keys and integers as values.

    Hint: I used a collections.defaultdict here. You can also use a collections.Counter.

    :param document: list of strings (words)
    :return: dict of IDF scores for each word
    """
    return Counter(document)

def idf(documents_containing_term: int, corpus_size: int) -> float:
    """
    Computes a single IDF score from a count of documents containing the term, and all documents,
    where IDF is defined as:
        t = term
        D = set of documents
        IDF(t, D) = log( |D| / |{d in D where d contains t}| ) + 1.0

        log base is e (sometimes written as ln)
        Note that we add 1.01 to all IDF scores, as smoothing. The smoothing could be performed in many other ways.

    :param documents_containing_term: A count of documents that contains the term
    :param corpus_size: Number of documents in the corpus
    """
    return log( corpus_size / documents_containing_term ) + 1.0

def calculate_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """
    Method takes the corpus (list of documents), and returns an Inverse Document Frequency dictionary.
    The function should return a dict from words to an IDF score.

    :param corpus: list of list of strings (words)
    :return: dict of word ->idf, i.e. idf scores for each word in the corpus
    """
    a = {}
    corpus_size = len(corpus)
    word_count = Counter(chain.from_iterable(set(x) for x in corpus))
    for k, v in word_count.items():
    	a[k] = idf(v, corpus_size)
    return a

def calculate_tfidf(tf: Dict[str, int], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates TF-IDF for a term frequency vector and an IDF vector.

    :param tf: dictionary of words and TFs
    :param idf: dictionary of words and IDFs
    :return: dictionary of words and TF-IDFs
    """
    a = {}
    rfset = set(tf)
    idfset = set(idf)
    for word in rfset.intersection(idfset):
    	a[word] = tf[word] * idf[word]
    for word in (idfset - rfset):
   	    a[word] = 0
    return a

def term2indices(corpus: List[List[str]]) -> Dict[str, int]:
    """
    Take the corpus (a list of documents), get the vocabulary and sort them alphabetically.
    Returns a word to index mapper as a dict. An index is a pointer to a position in a list/vector.

    :param corpus: list of list of strings (words)
    :return: a dict with strings as keys an indices (int) as values
    """
    a = {}
    word_count = Counter(chain.from_iterable(set(x) for x in corpus))
    index = 0
    for k, v in sorted(word_count.items()):
    	a[k] = index
    	index = index+1
    return a

def create_tfidf_vector(tfidf: Dict[str, float], word2index: Dict[str, int]) -> List[float]:
    """
    Take a document as TF-IDF scores, a word2index dict, and return a vector (list of floats)

    :param tfidf: TF-IDF scores for a document
    :param word2index:
    :return: TF-IDF vector as a list of floats
    """
    a = []
    for k in word2index:
    	a.append(tfidf[k])
    return a

def cosine_similarity(v1: List[float], v2: List[float]):
    """
    Computes the Cosine similarity between two vectors.
    Have a look at https://en.wikipedia.org/wiki/Cosine_similarity for the formula.

    :param v1: A vector
    :param v2: Another vector
    :return: the Cosine similarity as a float
    """
    ai2 = 0
    bi2 = 0
    sup_term = 0
    inf_term = 0
    if (len(v1) > len(v2)):
    	while len(v1) > len(v2):
    		v2.append(0)
    if (len(v2) > len(v1)):
    	while len(v2) > len(v1):
    		v1.append(0)
    for i in range(0,len(v1)):
    	sup_term = sup_term + (v1[i]*v2[i])
    	ai2 = ai2 + v1[i]**2
    	bi2 = bi2 + v2[i]**2
    inf_term = sqrt(ai2 * bi2)
    return sup_term / inf_term

def get_most_similar_pair(tfidfs: List[List[float]]) -> Tuple[float, int, int]:
    """
    Get the most similar pair of TF-IDF vectors, using Cosine as similarity metric.

    Hint: Use the cosine_similarity functio0n you have created.
          Also, you need to compare all-to-all here, and this will be slow.
          Try to make sure you don't compare the same documents multiple times.

    :param tfidfs: TF-IDF matrix (list of lists)
    :return: a tuple with (score, index_of_first_vector, _index_of_second_vector
    """
    vec1index = 0
    vec2index = 0
    winningscore = -1
    allvall = combinations(tfidfs, 2)
    for vector_pair in allvall:
    	score = cosine_similarity(vector_pair[0], vector_pair[1])
    	if score > winningscore:
    		winningscore = score
    		vec1index = tfidfs.index(vector_pair[0])
    		vec2index = tfidfs.index(vector_pair[1])
    return (winningscore, vec1index, vec2index)

def extract_contexts(documents: List[List[List[str]]], word2index: Dict[str, int], size: int) -> List[List[int]]:
    """
    Create a matrix (a list of lists) where:
    - the indices in the outer dimension are the indices of the target words
    - the indices in the inner dimension are the indices of the context words
    - the size parameter defines how many words we count before and after the target word
    - we don't do any padding, so for the first word of a sentence, we will only count the words after the target word.

    Note that we are not here dealing with a document-term matrix, but a term-term matrix.

    Hint: Initalize a matrix (list of lists) of zeroes,
          where both the outer and inner matrices have length = vocabulary size.

    :param documents: documents at sentence level
    :param word2index: word2index dict, in order to map the words to the indices
    :param size: defines the size of the context, so we take *size* words before and *size* words after the target word.
    :return: context term frequencies as a matrix of ints
    """
    a = [[0 for i in range(len(word2index.items()))] for j in range(len(word2index.items()))]
    for i in documents:
    	for j in i:
    		for w in range(0,len(j)):
    			for mov in range(1,size+1):
    				if (w+mov) <= (len(j)-1):
    					a[word2index[j[w]]][word2index[j[w+mov]]] += 1
    				if (w-mov) >= 0:
	    				a[word2index[j[w]]][word2index[j[w-mov]]] += 1
    return a

def reverse_word2index(word2index: Dict[str, int]) -> List[str]:
    """
    In order to map from indexes to words, we need to reverse the word2index mapping.
    This time, we'll do with a list of strings.
    :param word2index:
    :return: list of words
    """
    a = []
    for k, v in word2index.items():
    	a.append(k)
    return a

def idf_from_matrix(count_matrix: List[List[int]]):
    """
    Construct the IDF vector from a matrix instead of a list of dicts.
    :param count_matrix: A matrix of term-context counts
    :return: idf scores as a list of floats
    """
    aux = 0
    a = []
    for i in count_matrix:
    	for j in i:
    		if j != 0:
    			aux = aux + 1
    	a.append(idf(aux, len(count_matrix)))
    	aux = 0
    return a

def create_context_tfidf_vectors(contexts: List[List[int]], idf_vector: List[float]) -> List[List[float]]:
    """
    For each context (TF) in the context matrix, we need to scale with IDF.

    :param contexts:
    :param idf_vector:
    :return: a matrix of TF-IDF scores as floats
    """
    a = [[0 for i in range(len(contexts))] for j in range(len(contexts))]
    for i in range(len(contexts)):
    	for j in range(len(contexts)):
    		a[i][j] = contexts[i][j]*idf_vector[j]
    return a


def get_most_similar_to(idx: int, tfidfs: List[List[float]]) -> Tuple[float, int]:
    """
    Get the most similar index, with score, to the vector given by the ids.

    :param idx: index of the word we want to compare similarities to
    :param tfidfs: matrix of TF-IDF scores
    :return: a tuple of the similarity score and index of the most similar word
    """
    n = len(tfidfs)
    vec2index = 0
    winningscore = -1
    for i in range(0,n):
        if i != idx:
            score = cosine_similarity(tfidfs[i], tfidfs[idx])
            if score > winningscore:
                winningscore = score
                vec2index = i
    return (winningscore, vec2index)