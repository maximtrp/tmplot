import unittest
import pickle as pkl
from altair import LayerChart
from tomotopy import LDAModel
from src import tmplot as tm
from numpy import random, floating
from ipywidgets import VBox
from pandas import Series


class TestTmplot(unittest.TestCase):

    def setUp(self):
        self.tomotopy_model = LDAModel.load('tests/models/tomotopyLDA.model')
        with open('tests/models/gensimLDA.model', 'rb') as file:
            self.gensim_model = pkl.load(file)
        with open('tests/models/gensimLDA.corpus', 'rb') as file:
            self.gensim_corpus = pkl.load(file)
        with open('tests/models/btm_big.pickle', 'rb') as file:
            self.btm_model_big = pkl.load(file)
        with open('tests/models/btm_small.pickle', 'rb') as file:
            self.btm_model_small = pkl.load(file)

        self.phi = tm.get_phi(self.tomotopy_model)
        self.theta = tm.get_theta(self.tomotopy_model)

    def test_is_btmplus(self):
        self.assertTrue(tm._helpers._is_btmplus(self.btm_model_big))

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
        topics_coords = tm.prepare_coords(self.btm_model_big)
        self.assertTupleEqual(topics_coords.shape, (self.btm_model_big.topics_num_, 5))
        topics_coords = tm.prepare_coords(self.btm_model_small)
        self.assertTupleEqual(topics_coords.shape, (self.btm_model_small.topics_num_, 5))

    def test_get_topics_scatter(self):
        topics_dists = tm.get_topics_dist(self.phi)
        methods = ['tsne', 'sem', 'mds', 'lle', 'ltsa', 'isomap']
        topics_scatters = list(map(
            lambda method:
                tm.get_topics_scatter(topics_dists, self.theta, method=method),
            methods
        ))
        for scatter in topics_scatters:
            self.assertTupleEqual(scatter.shape, (self.tomotopy_model.k, 4))

    def test_get_topics_dist(self):
        methods = ["klb", "jsd", "jef", "hel", "bhat", "tv", "jac"]
        topics_dists = list(
            map(
                lambda method: tm.get_topics_dist(self.phi, method=method),
                methods)
        )
        for dist in topics_dists:
            self.assertTupleEqual(
                dist.shape,
                (self.tomotopy_model.k, self.tomotopy_model.k))

    def test_calc_topics_marg_probs(self):
        topic_marg_prob = tm.calc_topics_marg_probs(self.theta, 0)
        self.assertIsInstance(topic_marg_prob, floating)
        self.assertGreater(topic_marg_prob, 0)
        topics_marg_probs = tm.calc_topics_marg_probs(self.theta)
        self.assertIsInstance(topics_marg_probs, Series)
        self.assertEqual(topics_marg_probs.size, self.tomotopy_model.k)

    def test_calc_terms_marg_probs(self):
        term_marg_prob = tm.calc_terms_marg_probs(self.phi, 0)
        self.assertIsInstance(term_marg_prob, floating)
        self.assertGreater(term_marg_prob, 0)
        terms_marg_probs = tm.calc_terms_marg_probs(self.phi)
        self.assertIsInstance(terms_marg_probs, Series)
        self.assertEqual(terms_marg_probs.size, self.phi.index.size)

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

    def test_report(self):
        report = tm.report(
            self.tomotopy_model,
            docs=tm.get_docs(self.tomotopy_model),
            width=250)
        self.assertIsInstance(report, VBox)

    def test_entropy(self):
        entropy = tm.entropy(self.phi.T)
        self.assertGreater(entropy, 0)


if __name__ == '__main__':
    unittest.main()
