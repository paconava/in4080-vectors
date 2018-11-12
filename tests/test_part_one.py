import unittest
from typing import List

import nltk
from nltk.corpus import inaugural

import warnings

warnings.filterwarnings("ignore")


class TestPartOne(unittest.TestCase):
    """
    Test helper class, both to simplify code, and to make sure we expect the same results in both tests
    """

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

        self.documents_as_sentences = self.get_inaugural_docs()
        self.documents_as_words = [self.flatten_sentences(doc) for doc in self.documents_as_sentences]
        self.documents_as_raw_text = [' '.join(doc) for doc in self.documents_as_words]

    def assertCloseEnough(self, first, second):
        """
        Identity down to 6 decimal is good enough
        """
        self.assertAlmostEqual(first, second, places=6)

    @staticmethod
    def flatten_sentences(document: List[List[str]]) -> List[str]:
        """
        Method to take document as a list of sentences of words, and flatten it into a list of words
        :param document: list of list of strings (words)
        :return: list of strings
        """
        return [w for sent in document for w in sent]

    @staticmethod
    def get_inaugural_docs(download=False) -> List[List[List[str]]]:
        """
        Get the inaugural documents as a list (documents) of list (sentences) of list (sentence) of strings (words)
        :param download: If True, the corpus will be downloaded. Default=False
        :return:
        """
        if download:
            nltk.download('inaugural')
        return [[[w.lower() for w in sent] for sent in inaugural.sents(fileid)] for fileid in inaugural.fileids()]

    def test_documents_as_sentences(self):
        """Test that the first sentence of the first document is what we expect"""
        expected = ['fellow', '-', 'citizens', 'of', 'the', 'senate', 'and', 'of', 'the', 'house', 'of',
                    'representatives', ':']
        self.assertEqual(expected, self.documents_as_sentences[0][0])

    def test_documents_as_words(self):
        """Test that you flatten sentences to sentences to a list of words"""
        docs = self.documents_as_words

        self.assertEqual(len(docs[0]), 1538)

    def test_documents(self):
        """Test that you flatten sentences to sentences to a list of words"""
        docs = self.documents_as_words

        self.assertEqual(len(docs[0]), 1538)

    @property
    def idf_scores(self):
        return [
            ('the', 1.0),  # A very common word
            ('citizens', 1.0935260580108235),  # A common word
            ('life', 1.1967102942460544),  # A less common word
            ('vicissitudes', 3.4159137783010487),  # A very rare word
        ]

    @property
    def tfidf_scores(self):
        return [
            ('the', 116.0),
            ('citizens', 5.467630290054117),
            ('life', 1.1967102942460544),
            ('vicissitudes', 3.4159137783010487)
        ]

    @property
    def termcounts_in_first_doc(self):
        return [
            ('the', 116),
            ('citizens', 5),
            ('life', 1),
            ('vicissitudes', 1)
        ]

    @property
    def first_tfidf_vector(self):
        return {'length': 9070,
                'sum': 2682.135628064661,
                'citizens': 5.467630290054117}

    @property
    def cosine_similarities(self):
        return [1, 1, 0.7071067811865475, 0]

    @property
    def similarity_between_two_first_documents(self):
        return 0.6210000155106535

    @property
    def most_similar_documents(self):
        return {
            'doc1': '1841-Harrison.txt',
            'doc2': '1877-Hayes.txt',
            'score': 0.9353347825690648
        }

    @property
    def war(self):
        return {
            'count': 175,
            'context_terms': 662,
            'index': 8793,
            'tfidf_sum': 2567.679427692487,
            'most_similar_score': 0.7407871293777638
        }
