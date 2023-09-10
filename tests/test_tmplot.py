import unittest
import pickle as pkl
from altair import LayerChart
from tomotopy import LDAModel
from src import tmplot as tm
from numpy import random


class TestTmplot(unittest.TestCase):

    def setUp(self):
        self.tomotopy_model = LDAModel.load('tests/models/tomotopyLDA.model')
        with open('tests/models/gensimLDA.model', 'rb') as file:
            self.gensim_model = pkl.load(file)
        with open('tests/models/gensimLDA.corpus', 'rb') as file:
            self.gensim_corpus = pkl.load(file)

        self.phi = tm.get_phi(self.tomotopy_model)
        self.theta = tm.get_theta(self.tomotopy_model)

    def test_is_tomotopy(self):
        self.assertTrue(tm._helpers._is_tomotopy(self.tomotopy_model))

    def test_is_gensim(self):
        self.assertTrue(tm._helpers._is_gensim(self.gensim_model))

    def test_get_phi(self):
        phi = tm.get_phi(self.gensim_model)
        self.assertGreater(phi.shape[0], 0)
        self.assertGreater(self.phi.shape[0], 0)
        self.assertEqual(phi.shape[1], 15)
        self.assertEqual(self.phi.shape[1], 15)

    def test_get_theta(self):
        theta = tm.get_theta(self.gensim_model, self.gensim_corpus)
        self.assertEqual(theta.shape[0], 15)
        self.assertEqual(self.theta.shape[0], 15)
        self.assertGreater(theta.shape[1], 0)
        self.assertGreater(self.theta.shape[1], 0)

    def test_get_top_topic_words(self):
        top_words = tm.get_top_topic_words(self.phi)
        self.assertEqual(top_words.shape[0], 20)
        self.assertEqual(top_words.shape[1], self.phi.shape[1])

    def test_get_top_docs(self):
        docs = tm.get_docs(self.tomotopy_model)
        top_docs = tm.get_top_docs(docs, self.tomotopy_model, self.theta)
        self.assertEqual(top_docs.shape[0], 5)
        self.assertEqual(top_docs.shape[1], self.phi.shape[1])

    def test_get_relevant_terms(self):
        relevant_terms = tm.get_relevant_terms(self.phi, 0)
        self.assertEqual(relevant_terms.size, self.phi.shape[0])
        self.assertGreater(relevant_terms.iloc[0], relevant_terms.iloc[1])

    def test_prepare_coords(self):
        topics_coords = tm.prepare_coords(self.tomotopy_model)
        self.assertTupleEqual(topics_coords.shape, (self.tomotopy_model.k, 5))

    def test_plot_scatter_topics(self):
        topics_coords = tm.prepare_coords(self.tomotopy_model)
        chart = tm.plot_scatter_topics(
            topics_coords, size_col='size', label_col='label')
        self.assertIsInstance(chart, LayerChart)

    def test_get_stable_topics(self):
        models = [
            self.tomotopy_model, self.tomotopy_model, self.tomotopy_model,
            self.tomotopy_model]
        closest_topics, dists = tm.get_closest_topics(models)
        dists = random.normal(0, 0.10, dists.shape).__abs__()
        stable_topics, stable_dists = tm.get_stable_topics(
            closest_topics, dists, norm=False)

        self.assertTupleEqual(
            closest_topics.shape, (self.tomotopy_model.k, len(models)))
        self.assertTupleEqual(
            dists.shape, (self.tomotopy_model.k, len(models)))
        self.assertLessEqual(stable_topics.shape[0], self.tomotopy_model.k)
        self.assertLessEqual(stable_dists.shape[0], self.tomotopy_model.k)
        self.assertGreaterEqual(stable_topics.shape[0], 0)
        self.assertGreaterEqual(stable_dists.shape[0], 0)
        self.assertEqual(stable_topics.shape[1], len(models))
        self.assertEqual(stable_dists.shape[1], len(models))


if __name__ == '__main__':
    unittest.main()
