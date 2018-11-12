from scipy.sparse import csr_matrix

from assignment.part_one_with_libs import *
from tests.test_part_one import TestPartOne, inaugural


class TestPartOneWithLibraries(TestPartOne):

    def test_tf(self):
        """Test that you know how to count and store in a dict"""
        cv = get_count_vectorizer(self.documents_as_raw_text)
        tfs = cv.transform(self.documents_as_raw_text)

        def termcount_from_first_document(word):
            idx = cv.vocabulary_[word]
            return tfs[0, idx]

        for term, expected_count in self.termcounts_in_first_doc:
            self.assertEqual(expected_count, termcount_from_first_document(term))

    def test_idf(self):
        """Are the IDF calculations are performed correctly?"""
        cv = get_count_vectorizer(self.documents_as_raw_text)
        tfs = cv.transform(self.documents_as_raw_text)

        tfidf_vectorizer = get_tfidf_transformer(tfs)

        def idf_score(term):
            idx = cv.vocabulary_[term]
            return tfidf_vectorizer.idf_[idx]

        for term, expected_idf_score in self.idf_scores:
            self.assertCloseEnough(expected_idf_score, idf_score(term))

    def test_tfidf(self):
        """Test that the TFIDF scores are correct"""
        cv = get_count_vectorizer(self.documents_as_raw_text)
        tfs = cv.transform(self.documents_as_raw_text)

        tfidf_vectorizer = get_tfidf_transformer(tfs)
        tfidf_vectors = tfidf_vectorizer.transform(tfs)

        def tfidf_score(term):
            idx = cv.vocabulary_[term]
            return tfidf_vectors[0, idx]
        
        for term, expected_tfidf_score in self.tfidf_scores:
            self.assertCloseEnough(expected_tfidf_score, tfidf_score(term))


    def test_first_tfidf_vector(self):
        """Test that we can assemble the functions together and create a TFIDF vector with the right properties"""
        cv = get_count_vectorizer(self.documents_as_raw_text)
        tfs = cv.transform(self.documents_as_raw_text)

        tfidf_vectorizer = get_tfidf_transformer(tfs)
        tfidf_vector = tfidf_vectorizer.transform(tfs[0])

        self.assertIsInstance(tfidf_vector, csr_matrix)
        self.assertEqual(self.first_tfidf_vector['length'], tfidf_vector.shape[1])
        self.assertCloseEnough(self.first_tfidf_vector['sum'], tfidf_vector[0].sum())
        self.assertCloseEnough(self.first_tfidf_vector['citizens'], tfidf_vector[0, 1380])  # 'citizens'
        
    def test_cosine_similarity(self):
        """Cosine should be calculated correctly."""
        self.assertEqual(self.cosine_similarities[0], cosine_similarity([[1, 0]], [[1, 0]]))
        self.assertCloseEnough(self.cosine_similarities[1], cosine_similarity([[2, 0]], [[1, 0]]))
        self.assertCloseEnough(self.cosine_similarities[2], cosine_similarity([[1, 1]], [[1, 0]]))
        self.assertEqual(self.cosine_similarities[3], cosine_similarity([[1, 0]], [[0, 1]]))

    def test_cosine_similarity_on_first_document(self):
        """We should now be able to calculate the cosine similarity between the first and second speech"""
        cv = get_count_vectorizer(self.documents_as_raw_text)
        tfs = cv.transform(self.documents_as_raw_text)
        tfidf_vectorizer = get_tfidf_transformer(tfs)
        tfidf_vectors = tfidf_vectorizer.transform(tfs)

        similarity = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

        self.assertCloseEnough(similarity, self.similarity_between_two_first_documents)

    def test_most_similar_documents(self):
        """Now, we have all necesssary parts to find the most similar documents in a list of TFIDF vectors."""

        cv = get_count_vectorizer(self.documents_as_raw_text)
        tfs = cv.transform(self.documents_as_raw_text)
        tfidf_vectorizer = get_tfidf_transformer(tfs)
        tfidf_vectors = tfidf_vectorizer.transform(tfs)

        score, speech1, speech2 = get_most_similar_pair(tfidf_vectors)
        self.assertCloseEnough(score, self.most_similar_documents['score'])
        self.assertEqual(inaugural.fileids()[speech1], self.most_similar_documents['doc1'])
        self.assertEqual(inaugural.fileids()[speech2], self.most_similar_documents['doc2'])

    def test_extract_contexts(self):
        """Check that we are able to extract and count the contexts of terms"""
        cv = get_count_vectorizer(self.documents_as_raw_text)

        tfs = cv.transform(self.documents_as_raw_text)

        # Let's see how often the term 'vicissitudes' appears
        vicissitudes_count = sum([tf[0, cv.vocabulary_['vicissitudes']] for tf in tfs])
        self.assertEqual(5, vicissitudes_count)

        term2index = cv.vocabulary_

        contexts = extract_contexts(self.documents_as_sentences, term2index, 2)

        # Now that we have seen 5 occurrences, we should expect to see 5 * 4 context terms,
        # given that the term does not occur in the start or end of the sentence.
        # And in fact, it does, 'vicissitudes' is the second last term in a sentence.
        # So we only see 19 context terms for the 'vicissitudes'
        self.assertEqual(19, sum(contexts[term2index['vicissitudes']]))

        # And the term 'the' occurs 2 times in the context of 'vicissitudes'
        self.assertEqual(2, contexts[term2index['vicissitudes']][term2index['the']])

    def test_most_similar_term(self):
        """Check that we are able to find the most similar term given term-context TFIDFS"""
        cv = get_count_vectorizer(self.documents_as_raw_text)
        tfs = cv.transform(self.documents_as_raw_text)

        # Let's see how often the term 'war' appears
        war = self.war
        term2index = cv.vocabulary_

        term_count = sum([tf[0, term2index['war']] for tf in tfs])
        self.assertEqual(war['count'], term_count)
        # Now, let's construct TFIDF vectors for all terms, with size 2.
        # This contains quite a few steps:

        # 1. Extract the contexts as a V*V matrix of ints, where V is the length of the vocabulary
        contexts = extract_contexts(self.documents_as_sentences, term2index, 2)
        self.assertEqual(war['context_terms'], sum(contexts[term2index['war']]))

        # 2. Construct a reverse mapping of term2index, index2term, as a list of strings
        #    Secondly, we need a method to convert the idf scores to a list of floats
        #    This time, we'll compute the idf scores from the term context matrix
        index2term = reverse_word2index(term2index)
        self.assertEqual(war['index'], term2index['war'])
        self.assertEqual('war', index2term[war['index']])
        # 3. We can skip the explicit IDF vector...

        # 4. Create a TFIDF vector for all contexts
        #    Note that we need to supply the IDF scores here explicitly,
        #    as we want the IDF scores calculated on basis of the documents
        tfidf_transformer = get_tfidf_transformer(contexts)
        tfidfs = tfidf_transformer.transform(contexts)

        self.assertCloseEnough(war['tfidf_sum'], tfidfs[term2index['war']].sum())

        # 5. Use cosine to find the closest term to 'war'
        best_similarity, most_similar_idx = get_most_similar_to(term2index['war'], tfidfs)

        self.assertCloseEnough(war['most_similar_score'], best_similarity)

        # 6. Finally, print the most similar word to 'war
        most_similar_to_war = index2term[most_similar_idx]  # <- Replace this
        print(most_similar_to_war)