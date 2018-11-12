from assignment.part_one_no_libs import *
from tests.test_part_one import TestPartOne, inaugural


class TestsPartOneNoLibraries(TestPartOne):

    def test_tf(self):
        """Test that you know how to count and store in a dict"""
        tfs = [calculate_tf(bow) for bow in self.documents_as_words]
        first_speech = tfs[0]

        for word, expected_count in self.termcounts_in_first_doc:
            self.assertEqual(expected_count, first_speech[word])

    def test_single_idf(self):
        """Are the IDF calculations correct??"""

        self.assertCloseEnough(1.0, idf(1, 1))
        self.assertCloseEnough(1.6931471805599454, idf(1, 2))
        self.assertCloseEnough(3.302585092994046, idf(1, 10))

    def test_idf(self):
        """Is the IDF vector correct?"""
        idfs = calculate_idf(self.documents_as_words)

        for word, expected_idf_score in self.idf_scores:
            self.assertCloseEnough(expected_idf_score, idfs[word])

    def test_tfidf(self):
        """Test that the TFIDF scores are correct"""
        docs = self.documents_as_words
        idfs = calculate_idf(docs)
        tfs = [calculate_tf(bow) for bow in docs]
        tfidfs = [calculate_tfidf(tf, idfs) for tf in tfs]
        first_speech_tfidfs = tfidfs[0]

        for word, expected_tfidf_score in self.tfidf_scores:
            self.assertCloseEnough(expected_tfidf_score, first_speech_tfidfs[word])

    def test_word2index(self):
        """Test that we map correctly from words to indices"""
        docs = self.documents_as_words
        word2index = term2indices(docs)

        self.assertEqual(len(word2index), 9070)
        self.assertEqual(8071, word2index['the'])
        self.assertEqual(1380, word2index['citizens'])
        self.assertEqual(4861, word2index['life'])
        self.assertEqual(8703, word2index['vicissitudes'])

    def test_first_tfidf_vector(self):
        """Test that we can assemble the functions together and create a TFIDF vector with the right properties"""
        docs = self.documents_as_words
        tfs = [calculate_tf(doc) for doc in docs]
        idfs = calculate_idf(docs)
        tfidfs = [calculate_tfidf(tf, idfs) for tf in tfs]
        word2index = term2indices(docs)
        tfidf_vector = create_tfidf_vector(tfidfs[0], word2index)

        self.assertEqual(self.first_tfidf_vector['length'], len(tfidf_vector))
        self.assertCloseEnough(self.first_tfidf_vector['sum'], sum(tfidf_vector))
        self.assertCloseEnough(self.first_tfidf_vector['citizens'], tfidf_vector[1380])  # 'citizens'

    def test_cosine_similarity(self):
        """Cosine should be calculated correctly."""
        self.assertEqual(self.cosine_similarities[0], cosine_similarity([1, 0], [1, 0]))
        self.assertCloseEnough(self.cosine_similarities[1], cosine_similarity([2, 0], [1, 0]))
        self.assertCloseEnough(self.cosine_similarities[2], cosine_similarity([1, 1], [1, 0]))
        self.assertEqual(self.cosine_similarities[3], cosine_similarity([1, 0], [0, 1]))

    def test_cosine_similarity_on_first_document(self):
        """We should now be able to calculate the cosine similarity between the first and second speech"""
        docs = self.documents_as_words
        tfs = [calculate_tf(bow) for bow in docs]
        idfs = calculate_idf(docs)
        tfidfs = [calculate_tfidf(tf, idfs) for tf in tfs]
        word2index = term2indices(docs)

        first_tfidf_vector = create_tfidf_vector(tfidfs[0], word2index)
        second_tfidf_vector = create_tfidf_vector(tfidfs[1], word2index)
        similarity = cosine_similarity(first_tfidf_vector, second_tfidf_vector)

        self.assertCloseEnough(similarity, self.similarity_between_two_first_documents)

    def test_most_similar_documents(self):
        """Now, we have all necesssary parts to find the most similar documents in a list of TFIDF vectors."""
        docs = self.documents_as_words
        word2index = term2indices(docs)
        idf = calculate_idf(docs)
        tfs = [calculate_tf(bow) for bow in docs]
        tfidfs = [calculate_tfidf(tf, idf) for tf in tfs]
        tfidf_vectors = [create_tfidf_vector(tfidf, word2index) for tfidf in tfidfs]
        score, speech1, speech2 = get_most_similar_pair(tfidf_vectors)

        self.assertCloseEnough(score, self.most_similar_documents['score'])
        self.assertEqual(inaugural.fileids()[speech1], self.most_similar_documents['doc1'])
        self.assertEqual(inaugural.fileids()[speech2], self.most_similar_documents['doc2'])

    def test_extract_contexts(self):
        """Check that we are able to extract and count the contexts of words with different size"""
        word2index = term2indices(self.documents_as_words)

        # Let's see how often the word 'vicissitudes' appears
        word_counts = [calculate_tf(bow) for bow in self.documents_as_words]
        vicissitudes_count = sum([tf['vicissitudes'] for tf in word_counts])
        self.assertEqual(5, vicissitudes_count)
        
        contexts = extract_contexts(self.documents_as_sentences, word2index, 2)

        # Now that we have seen 5 occurrences, we should expect to see 5 * 4 context words,
        # given that the word does not occur in the start or end of the sentence.
        # And in fact, it does, 'vicissitudes' is the second last word in a sentence.
        # So we only see 19 context words for the 'vicissitudes'
        self.assertEqual(19, sum(contexts[word2index['vicissitudes']]))

        # And the word 'the' occurs 2 times in the context of 'vicissitudes'
        self.assertEqual(2, contexts[word2index['vicissitudes']][word2index['the']])

    def test_most_similar_term(self):
        """Check that we are able to find the most similar term given term-context TFIDFS"""
        term2index = term2indices(self.documents_as_words)
        word_counts = [calculate_tf(bow) for bow in self.documents_as_words]

        # Let's see how often the term 'war' appears
        war = self.war
        term_count = sum([tf['war'] for tf in word_counts])
        self.assertEqual(war['count'], term_count)

        # Now, let's construct TFIDF vectors for all words, with size 2.
        # This contains quite a few steps:

        # 1. Extract the contexts as a V*V matrix of ints, where V is the length of the vocabulary
        contexts = extract_contexts(self.documents_as_sentences, term2index, 2)
        self.assertEqual(war['context_terms'], sum(contexts[term2index['war']]))
        
        # 2. Construct a reverse mapping of term2index, index2word, as a list of strings
        # Secondly, we need a method to convert the idf scores to a list of floats
        # Here, we'll use the same idf scores as we used before for simplicity
        index2word = reverse_word2index(term2index)
        self.assertEqual(war['index'], term2index['war'])
        self.assertEqual('war', index2word[war['index']])

        # 3. Create an IDF vector (as opposed to a dict), from the term-context matrix
        idf_vector = idf_from_matrix(contexts)
        self.assertCloseEnough(4.615559317815981, idf_vector[war['index']])

        # 4. Create a TFIDF vector for all contexts
        tfidfs = create_context_tfidf_vectors(contexts, idf_vector)
        self.assertCloseEnough(war['tfidf_sum'], sum(tfidfs[term2index['war']]))

        # 5. Use cosine to find the closest term to 'war'
        best_similarity, most_similar_idx = get_most_similar_to(term2index['war'], tfidfs)

        self.assertCloseEnough(war['most_similar_score'], best_similarity)

        # 6. Finally, print the most similar word to 'war
        most_similar_to_war = index2word[most_similar_idx]  # <- Replace this
        print(most_similar_to_war)

