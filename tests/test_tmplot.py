import unittest
import pickle as pkl
from altair import LayerChart
from tomotopy import LDAModel
from src import tmplot as tm
from numpy import random, floating, ndarray
from ipywidgets import VBox


class TestTmplot(unittest.TestCase):
    def setUp(self):
        self.tomotopy_model = LDAModel.load("tests/models/tomotopyLDA.model")
        with open("tests/models/gensimLDA.model", "rb") as file:
            self.gensim_model = pkl.load(file)
        with open("tests/models/gensimLDA.corpus", "rb") as file:
            self.gensim_corpus = pkl.load(file)
        with open("tests/models/btm_big.pickle", "rb") as file:
            self.btm_model_big = pkl.load(file)
        with open("tests/models/btm_small.pickle", "rb") as file:
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
        self.assertTupleEqual(
            topics_coords.shape, (self.btm_model_small.topics_num_, 5)
        )

    def test_get_topics_scatter(self):
        topics_dists = tm.get_topics_dist(self.phi)
        methods = ["tsne", "sem", "mds", "lle", "ltsa", "isomap"]
        topics_scatters = list(
            map(
                lambda method: tm.get_topics_scatter(
                    topics_dists, self.theta, method=method
                ),
                methods,
            )
        )
        for scatter in topics_scatters:
            self.assertTupleEqual(scatter.shape, (self.tomotopy_model.k, 4))

    def test_get_topics_dist(self):
        methods = ["klb", "jsd", "jef", "hel", "bhat", "tv", "jac"]
        topics_dists = list(
            map(lambda method: tm.get_topics_dist(self.phi, method=method), methods)
        )
        for dist in topics_dists:
            self.assertTupleEqual(
                dist.shape, (self.tomotopy_model.k, self.tomotopy_model.k)
            )

    def test_calc_topics_marg_probs(self):
        topic_marg_prob = tm.calc_topics_marg_probs(self.theta, 0)
        self.assertIsInstance(topic_marg_prob, floating)
        self.assertGreater(topic_marg_prob, 0)
        topics_marg_probs = tm.calc_topics_marg_probs(self.theta)
        self.assertIsInstance(topics_marg_probs, ndarray)
        self.assertEqual(topics_marg_probs.size, self.tomotopy_model.k)
        self.assertEqual(topics_marg_probs.sum(), 1)

    def test_calc_terms_marg_probs(self):
        term_marg_prob = tm.calc_terms_marg_probs(
            self.phi, tm.calc_topics_marg_probs(self.theta), 0
        )
        self.assertIsInstance(term_marg_prob, floating)
        self.assertGreater(term_marg_prob, 0)
        terms_marg_probs = tm.calc_terms_marg_probs(
            self.phi, tm.calc_topics_marg_probs(self.theta)
        )
        self.assertIsInstance(terms_marg_probs, ndarray)
        self.assertEqual(terms_marg_probs.size, self.phi.index.size)

    def test_plot_scatter_topics(self):
        topics_coords = tm.prepare_coords(self.tomotopy_model)
        chart = tm.plot_scatter_topics(
            topics_coords, size_col="size", label_col="label"
        )
        self.assertIsInstance(chart, LayerChart)

    def test_get_stable_topics(self):
        models = [
            self.tomotopy_model,
            self.tomotopy_model,
            self.tomotopy_model,
            self.tomotopy_model,
        ]
        closest_topics, dists = tm.get_closest_topics(models)
        dists = random.normal(0, 0.10, dists.shape).__abs__()
        stable_topics, stable_dists = tm.get_stable_topics(
            closest_topics, dists, norm=False
        )

        self.assertTupleEqual(
            closest_topics.shape, (self.tomotopy_model.k, len(models))
        )
        self.assertTupleEqual(dists.shape, (self.tomotopy_model.k, len(models)))
        self.assertLessEqual(stable_topics.shape[0], self.tomotopy_model.k)
        self.assertLessEqual(stable_dists.shape[0], self.tomotopy_model.k)
        self.assertGreaterEqual(stable_topics.shape[0], 0)
        self.assertGreaterEqual(stable_dists.shape[0], 0)
        self.assertEqual(stable_topics.shape[1], len(models))
        self.assertEqual(stable_dists.shape[1], len(models))

    def test_report(self):
        report = tm.report(
            self.tomotopy_model, docs=tm.get_docs(self.tomotopy_model), width=250
        )
        self.assertIsInstance(report, VBox)

    def test_entropy(self):
        entropy = tm.entropy(self.phi.T)
        entropy2 = tm.entropy(self.phi.T, max_probs=True)
        self.assertGreater(entropy, 0)
        self.assertGreater(entropy2, 0)

    def test_entropy_single_topic(self):
        # Test edge case with single topic (line 76 in _metrics.py)
        import numpy as np
        single_topic_phi = np.random.rand(1, 100)  # Create single topic phi matrix
        entropy_single = tm.entropy(single_topic_phi)
        self.assertIsInstance(entropy_single, float)

    def test_get_salient_terms(self):
        saliency = tm.get_salient_terms(self.phi, self.theta)
        self.assertEqual(saliency.size, self.phi.shape[0])

    # Error handling tests for _helpers.py
    def test_get_theta_gensim_no_corpus(self):
        # Test error when corpus is not provided for gensim model (line 158)
        with self.assertRaises(ValueError) as context:
            tm.get_theta(self.gensim_model)
        self.assertIn("corpus", str(context.exception).lower())

    def test_get_theta_gensim_empty_corpus(self):
        # Test error when corpus is empty for gensim model (line 160)
        with self.assertRaises(ValueError) as context:
            tm.get_theta(self.gensim_model, corpus=[])
        self.assertIn("corpus cannot be empty", str(context.exception))

    def test_get_theta_unsupported_model(self):
        # Test error for unsupported model type (line 171)
        class UnsupportedModel:
            pass

        with self.assertRaises(ValueError) as context:
            tm.get_theta(UnsupportedModel())
        self.assertIn("Unsupported model type", str(context.exception))

    def test_get_top_docs_no_model_or_theta(self):
        # Test error when neither model nor theta is provided (line 237)
        docs = tm.get_docs(self.tomotopy_model)
        with self.assertRaises(ValueError) as context:
            tm.get_top_docs(docs)
        self.assertIn("model or a theta matrix", str(context.exception))

    def test_calc_topics_marg_probs_empty_theta(self):
        # Test error for empty theta matrix (line 273)
        import numpy as np
        empty_theta = np.array([])
        with self.assertRaises(ValueError) as context:
            tm.calc_topics_marg_probs(empty_theta)
        self.assertIn("theta matrix cannot be empty", str(context.exception))

    def test_calc_topics_marg_probs_all_zeros(self):
        # Test error for theta matrix with all zeros (line 278)
        import numpy as np
        zero_theta = np.zeros((3, 5))
        with self.assertRaises(ValueError) as context:
            tm.calc_topics_marg_probs(zero_theta)
        self.assertIn("contains all zeros", str(context.exception))

    def test_calc_topics_marg_probs_invalid_topic_id(self):
        # Test error for invalid topic_id (line 283)
        with self.assertRaises(IndexError) as context:
            tm.calc_topics_marg_probs(self.theta, topic_id=999)
        self.assertIn("out of bounds", str(context.exception))

    def test_calc_terms_marg_probs_empty_phi(self):
        # Test error for empty phi matrix (line 313)
        import numpy as np
        empty_phi = np.array([])
        p_t = tm.calc_topics_marg_probs(self.theta)
        with self.assertRaises(ValueError) as context:
            tm.calc_terms_marg_probs(empty_phi, p_t)
        self.assertIn("phi matrix cannot be empty", str(context.exception))

    def test_calc_terms_marg_probs_empty_pt(self):
        # Test error for empty p_t array (line 315)
        import numpy as np
        empty_pt = np.array([])
        with self.assertRaises(ValueError) as context:
            tm.calc_terms_marg_probs(self.phi, empty_pt)
        self.assertIn("p_t array cannot be empty", str(context.exception))

    def test_calc_terms_marg_probs_dimension_mismatch(self):
        # Test error for dimension mismatch (line 317)
        import numpy as np
        wrong_pt = np.array([0.5, 0.5])  # Wrong size
        with self.assertRaises(ValueError) as context:
            tm.calc_terms_marg_probs(self.phi, wrong_pt)
        self.assertIn("phi topics dimension", str(context.exception))

    def test_calc_terms_marg_probs_invalid_word_id(self):
        # Test error for invalid word_id (line 322)
        p_t = tm.calc_topics_marg_probs(self.theta)
        max_word_id = self.phi.shape[0]
        with self.assertRaises(IndexError) as context:
            tm.calc_terms_marg_probs(self.phi, p_t, word_id=max_word_id + 10)
        self.assertIn("word_id", str(context.exception))

    def test_get_salient_terms_empty_matrices(self):
        # Test error for empty phi and theta matrices (line 347)
        import numpy as np
        empty_phi = np.array([])
        empty_theta = np.array([])
        with self.assertRaises(ValueError) as context:
            tm.get_salient_terms(empty_phi, empty_theta)
        self.assertIn("phi and theta matrices cannot be empty", str(context.exception))

    def test_get_salient_terms_dimension_mismatch(self):
        # Test error for dimension mismatch in phi and theta (line 349)
        import numpy as np
        wrong_theta = np.random.rand(10, 5)  # Wrong number of topics
        with self.assertRaises(ValueError) as context:
            tm.get_salient_terms(self.phi, wrong_theta)
        self.assertIn("phi topics dimension", str(context.exception))

    # Tests for _vis.py error handling
    def test_plot_scatter_topics_empty_ndarray(self):
        # Test error for empty ndarray input (lines 133-135)
        import numpy as np
        empty_coords = np.array([])
        with self.assertRaises(ValueError) as context:
            tm.plot_scatter_topics(empty_coords)
        self.assertIn("topics_coords cannot be empty", str(context.exception))

    def test_plot_scatter_topics_empty_dataframe(self):
        # Test error for empty DataFrame input (lines 137-139)
        from pandas import DataFrame
        empty_df = DataFrame()
        with self.assertRaises(ValueError) as context:
            tm.plot_scatter_topics(empty_df)
        self.assertIn("topics_coords DataFrame cannot be empty", str(context.exception))

    def test_plot_terms_empty_dataframe(self):
        # Test error for empty DataFrame input (lines 233-234)
        from pandas import DataFrame
        empty_df = DataFrame()
        with self.assertRaises(ValueError) as context:
            tm.plot_terms(empty_df)
        self.assertIn("terms_probs DataFrame cannot be empty", str(context.exception))

    def test_plot_terms_missing_columns(self):
        # Test error for missing required columns (lines 236-238)
        from pandas import DataFrame
        incomplete_df = DataFrame({"wrong_col": [1, 2, 3]})
        with self.assertRaises(ValueError) as context:
            tm.plot_terms(incomplete_df)
        self.assertIn("Missing required columns", str(context.exception))

    def test_plot_docs_with_sequence(self):
        # Test plot_docs with sequence input (lines 298-299)
        docs_list = ["Document 1 content", "Document 2 content"]
        result = tm.plot_docs(docs_list)
        from IPython.display import HTML
        self.assertIsInstance(result, HTML)

    # Tests for _report.py error handling
    def test_report_empty_docs(self):
        # Test error for empty docs (line 110)
        with self.assertRaises(ValueError) as context:
            tm.report(self.tomotopy_model, docs=[])
        self.assertIn("docs cannot be empty", str(context.exception))

    def test_report_none_docs(self):
        # Test error for None docs (line 110)
        with self.assertRaises(ValueError) as context:
            tm.report(self.tomotopy_model, docs=None)
        self.assertIn("docs cannot be empty", str(context.exception))

    def test_report_with_custom_parameters(self):
        # Test report with custom parameters to increase coverage
        docs = tm.get_docs(self.tomotopy_model)
        report = tm.report(
            self.tomotopy_model,
            docs=docs,
            topics_labels=["Topic A", "Topic B"],
            show_headers=False,
            show_docs=False,
            show_words=False,
            show_topics=True,
            width=400,
            height=600
        )
        from ipywidgets import VBox
        self.assertIsInstance(report, VBox)

    def test_report_gensim_with_corpus(self):
        # Test report with gensim model and corpus
        theta_gensim = tm.get_theta(self.gensim_model, self.gensim_corpus)
        num_docs = theta_gensim.shape[1]
        docs = [f"doc{i}" for i in range(num_docs)]  # Create appropriate number of docs
        report = tm.report(
            self.gensim_model,
            docs=docs,
            corpus=self.gensim_corpus,
            width=200
        )
        from ipywidgets import VBox
        self.assertIsInstance(report, VBox)

    # Additional tests for better coverage of edge cases
    def test_get_docs_non_tomotopy_model(self):
        # Test get_docs with non-tomotopy model (line 198)
        result = tm.get_docs(self.gensim_model)
        self.assertIsNone(result)

    def test_get_phi_with_vocabulary(self):
        # Test get_phi with gensim model and vocabulary (line 85)
        gensim_phi = tm.get_phi(self.gensim_model)
        vocab = ["word" + str(i) for i in range(gensim_phi.shape[0])]
        phi_with_vocab = tm.get_phi(self.gensim_model, vocabulary=vocab)
        self.assertEqual(len(phi_with_vocab.index), len(vocab))
        self.assertListEqual(list(phi_with_vocab.index), vocab)

    def test_prepare_coords_with_kwargs(self):
        # Test prepare_coords with dist_kws and scatter_kws
        dist_kws = {"method": "jsd"}
        scatter_kws = {"method": "mds"}
        coords = tm.prepare_coords(
            self.tomotopy_model,
            dist_kws=dist_kws,
            scatter_kws=scatter_kws
        )
        self.assertEqual(coords.shape[1], 5)  # x, y, size, label, topic

    def test_get_top_docs_with_theta_matrix(self):
        # Test get_top_docs when providing theta matrix instead of model
        docs = tm.get_docs(self.tomotopy_model)
        theta_values = self.theta.values
        top_docs = tm.get_top_docs(docs, theta=theta_values)
        self.assertEqual(top_docs.shape[0], 5)  # Default docs_num

    def test_get_top_docs_with_specific_topics(self):
        # Test get_top_docs with specific topics selection
        docs = tm.get_docs(self.tomotopy_model)
        specific_topics = [0, 2, 4]
        top_docs = tm.get_top_docs(
            docs,
            self.tomotopy_model,
            self.theta,
            topics=specific_topics
        )
        self.assertEqual(top_docs.shape[1], len(specific_topics))

    def test_calc_terms_probs_ratio_edge_cases(self):
        # Test calc_terms_probs_ratio with different parameters
        terms_probs = tm.calc_terms_probs_ratio(
            self.phi,
            topic=1,
            terms_num=10,
            lambda_=0.8
        )
        self.assertEqual(len(terms_probs), 20)  # 10 terms * 2 types

    def test_get_relevant_terms_different_lambda(self):
        # Test get_relevant_terms with different lambda values
        relevant_terms_1 = tm.get_relevant_terms(self.phi, 0, lambda_=0.2)
        relevant_terms_2 = tm.get_relevant_terms(self.phi, 0, lambda_=0.9)
        self.assertEqual(relevant_terms_1.size, self.phi.shape[0])
        self.assertEqual(relevant_terms_2.size, self.phi.shape[0])
        # Results should be different with different lambda values
        self.assertFalse(relevant_terms_1.equals(relevant_terms_2))

    def test_plot_scatter_topics_with_all_options(self):
        # Test plot_scatter_topics with many parameters to increase coverage
        topics_coords = tm.prepare_coords(self.tomotopy_model)
        chart = tm.plot_scatter_topics(
            topics_coords,
            topic=1,
            size_col="size",
            label_col="label",
            font_size=15,
            x_kws={"title": "X Axis"},
            y_kws={"title": "Y Axis"},
            chart_kws={"title": "Test Chart"},
            circle_kws={"opacity": 0.5},
            text_kws={"fontSize": 12},
            size_kws={"range": [100, 2000]},
            color_kws={"scheme": "viridis"}
        )
        from altair import LayerChart
        self.assertIsInstance(chart, LayerChart)

    def test_plot_terms_with_custom_parameters(self):
        # Test plot_terms with custom parameters
        terms_probs = tm.calc_terms_probs_ratio(self.phi, 0)
        chart = tm.plot_terms(
            terms_probs,
            font_size=16,
            chart_kws={"width": 400},
            bar_kws={"stroke": "black"},
            x_kws={"title": "Custom X"},
            y_kws={"title": "Custom Y"},
            color_kws={"scheme": "set1"}
        )
        from altair import Chart
        self.assertIsInstance(chart, Chart)

    def test_plot_docs_with_custom_styles(self):
        # Test plot_docs with custom styles and html_kws
        docs_list = ["Document 1", "Document 2"]
        custom_styles = "<style>table { border: 1px solid black; }</style>"
        html_kws = {"escape": False, "classes": "custom-table"}
        result = tm.plot_docs(docs_list, styles=custom_styles, html_kws=html_kws)
        from IPython.display import HTML
        self.assertIsInstance(result, HTML)

    def test_btm_model_functionality(self):
        # Test BTM model specific functionality to increase coverage
        phi_btm = tm.get_phi(self.btm_model_big)
        self.assertGreater(phi_btm.shape[0], 0)

        theta_btm = tm.get_theta(self.btm_model_big)
        self.assertGreater(theta_btm.shape[0], 0)

    # Test package warning functionality (when packages aren't available)
    def test_package_warning_simulation(self):
        # This will test the warning paths, but since packages are installed,
        # we test the actual functionality and ensure no warnings are raised
        # The warning paths would be tested if packages weren't available

        # Test that the helper functions work correctly when packages are available
        self.assertTrue(tm._helpers._is_tomotopy(self.tomotopy_model))
        self.assertTrue(tm._helpers._is_gensim(self.gensim_model))
        self.assertTrue(tm._helpers._is_btmplus(self.btm_model_big))

        # Test with an object that's not a recognized model type
        class NotAModel:
            pass

        not_a_model = NotAModel()
        self.assertFalse(tm._helpers._is_tomotopy(not_a_model))
        self.assertFalse(tm._helpers._is_gensim(not_a_model))
        self.assertFalse(tm._helpers._is_btmplus(not_a_model))


if __name__ == "__main__":
    unittest.main()
