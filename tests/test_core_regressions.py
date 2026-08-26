from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src import tmplot as tm


def test_kl_distance_keeps_zero_support_contribution():
    distance = tm._distance._dist_klb(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.isfinite(distance)
    assert distance > 100


@pytest.mark.parametrize("function", [tm._distance._dist_sklb, tm._distance._dist_jef])
def test_symmetric_divergences_keep_zero_support_contribution(function):
    assert function(np.array([1.0, 0.0]), np.array([0.0, 1.0])) > 100


def test_unknown_distance_method_has_clear_error():
    phi = np.array([[0.7, 0.3], [0.3, 0.7]])
    with pytest.raises(ValueError, match="Unknown distance method"):
        tm.get_topics_dist(phi, method="unknown")


def test_unknown_scatter_method_has_clear_error():
    with pytest.raises(ValueError, match="Unknown scatter method"):
        tm.get_topics_scatter(np.zeros((2, 2)), np.ones((2, 1)), method="unknown")


@pytest.mark.parametrize(
    "distances,theta,message",
    [
        (np.zeros((2, 3)), np.ones((2, 1)), "square"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), np.ones((2, 1)), "finite"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), np.ones((2, 1)), "symmetric"),
        (np.zeros((2, 2)), np.ones((3, 1)), "theta topics"),
    ],
)
def test_scatter_validates_matrix_contract(distances, theta, message):
    with pytest.raises(ValueError, match=message):
        tm.get_topics_scatter(distances, theta)


@pytest.mark.parametrize("method", ["tsne", "sem", "mds", "lle", "ltsa", "isomap"])
def test_scatter_supports_two_topics(method):
    result = tm.get_topics_scatter(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.ones((2, 1)),
        method=method,
    )
    assert result[["x", "y"]].shape == (2, 2)


def test_spectral_embedding_receives_affinity_not_distance():
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    theta = np.ones((3, 1))
    transformer = Mock()
    transformer.fit_transform.return_value = np.zeros((3, 2))
    with patch("src.tmplot._distance.SpectralEmbedding", return_value=transformer):
        tm.get_topics_scatter(distances, theta, method="sem")
    affinity = transformer.fit_transform.call_args.args[0]
    assert np.allclose(np.diag(affinity), 1.0)
    assert affinity[0, 1] < affinity[0, 0]


def test_tsne_uses_precomputed_distances():
    transformer = Mock()
    transformer.fit_transform.return_value = np.zeros((3, 2))
    with patch("src.tmplot._distance.TSNE", return_value=transformer) as constructor:
        tm.get_topics_scatter(np.zeros((3, 3)), np.ones((3, 1)), method="tsne")
    assert constructor.call_args.kwargs["metric"] == "precomputed"
    assert constructor.call_args.kwargs["init"] == "random"


def test_lle_uses_coordinates_reconstructed_from_distances():
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    transformer = Mock()
    transformer.fit_transform.return_value = np.zeros((3, 2))
    with patch(
        "src.tmplot._distance.LocallyLinearEmbedding", return_value=transformer
    ) as constructor:
        tm.get_topics_scatter(distances, np.ones((3, 1)), method="lle")
    transformed_input = transformer.fit_transform.call_args.args[0]
    assert transformed_input.shape[0] == 3
    assert transformed_input.shape[1] < distances.shape[1]
    assert constructor.call_args.kwargs["n_neighbors"] == 2


def test_top_topic_words_accepts_numpy_topic_indices():
    phi = pd.DataFrame({0: [0.8, 0.2], 1: [0.1, 0.9]}, index=["a", "b"])
    result = tm.get_top_topic_words(phi, topics_idx=np.array([1]))
    assert result.columns.tolist() == [1]


def test_get_top_docs_derives_theta_from_model():
    with patch(
        "src.tmplot._helpers.get_theta", return_value=pd.DataFrame([[0.1, 0.9]])
    ):
        result = tm.get_top_docs(["low", "high"], model=object(), docs_num=1)
    assert result.iloc[0, 0] == "high"


def test_get_top_docs_preserves_explicit_theta():
    theta = np.array([[0.9, 0.1]])
    with patch("src.tmplot._helpers.get_theta") as get_theta:
        result = tm.get_top_docs(
            ["high", "low"], model=object(), theta=theta, docs_num=1
        )
    get_theta.assert_not_called()
    assert result.iloc[0, 0] == "high"


def test_get_top_docs_validates_document_count():
    with pytest.raises(ValueError, match="docs length"):
        tm.get_top_docs(["one"], theta=np.ones((1, 2)))


def test_relevant_terms_supports_ndarray_and_topic_prevalence():
    phi = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = tm.get_relevant_terms(phi, 0, p_t=np.array([0.99, 0.01]))
    assert isinstance(result, pd.Series)
    assert result.index[0] == 0


def test_term_probabilities_use_topic_prevalence():
    phi = pd.DataFrame([[0.9, 0.1], [0.1, 0.9]], index=["a", "b"])
    result = tm.calc_terms_probs_ratio(phi, 0, p_t=np.array([0.99, 0.01]))
    marginal = result[result["Type"].str.startswith("Marginal")].set_index("Terms")
    assert marginal.loc["a", "Probability"] == pytest.approx(0.892)
    assert marginal.loc["b", "Probability"] == pytest.approx(0.108)


def test_probability_helpers_validate_distributions():
    with pytest.raises(ValueError, match="2D"):
        tm.calc_topics_marg_probs(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite non-negative"):
        tm.calc_topics_marg_probs(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="finite non-negative"):
        tm.calc_terms_marg_probs(np.array([[-1.0]]), np.array([1.0]))
    with pytest.raises(ValueError, match="positive total"):
        tm.calc_terms_marg_probs(np.array([[1.0]]), np.array([0.0]))
    with pytest.raises(ValueError, match="lambda_"):
        tm.get_relevant_terms(np.array([[1.0]]), 0, lambda_=1.1)


def test_term_marginals_normalize_prevalence_weights():
    result = tm.calc_terms_marg_probs(
        np.array([[0.8, 0.2], [0.2, 0.8]]), np.array([9.0, 1.0])
    )
    assert result.tolist() == pytest.approx([0.74, 0.26])


def test_saliency_matches_vectorized_definition():
    phi = np.array([[0.8, 0.2], [0.2, 0.8]])
    theta = np.array([[0.9, 0.9], [0.1, 0.1]])
    result = tm.get_salient_terms(phi, theta)
    p_t = tm.calc_topics_marg_probs(theta)
    p_w = tm.calc_terms_marg_probs(phi, p_t)
    p_tw = phi * p_t / p_w[:, None]
    expected = p_w * np.sum(p_tw * np.log(p_tw / p_t), axis=1)
    assert np.allclose(result, expected)


def test_get_phi_accepts_numpy_vocabulary_and_validates_length():
    model = Mock()
    model.get_topics.return_value = np.array([[0.8, 0.2], [0.2, 0.8]])
    with (
        patch("src.tmplot._helpers._is_tomotopy", return_value=False),
        patch("src.tmplot._helpers._is_gensim", return_value=True),
    ):
        result = tm.get_phi(model, vocabulary=np.array(["a", "b"]))
        with pytest.raises(ValueError, match="vocabulary length"):
            tm.get_phi(model, vocabulary=["a"])
    assert result.index.tolist() == ["a", "b"]


def test_get_phi_rejects_unsupported_model():
    with (
        patch("src.tmplot._helpers._is_tomotopy", return_value=False),
        patch("src.tmplot._helpers._is_gensim", return_value=False),
        patch("src.tmplot._helpers._is_btmplus", return_value=False),
        patch("src.tmplot._helpers.tomotopy_installed", True),
        patch("src.tmplot._helpers.gensim_installed", True),
        patch("src.tmplot._helpers.bitermplus_installed", True),
    ):
        with pytest.raises(ValueError, match="Unsupported model type"):
            tm.get_phi(object())


def test_optional_adapter_checks_do_not_warn_for_unrelated_objects():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tm._helpers._is_gensim(object())
        tm._helpers._is_tomotopy(object())
        tm._helpers._is_btmplus(object())
    assert caught == []


def test_unsupported_model_warns_about_missing_adapters():
    with (
        patch("src.tmplot._helpers._is_tomotopy", return_value=False),
        patch("src.tmplot._helpers._is_gensim", return_value=False),
        patch("src.tmplot._helpers._is_btmplus", return_value=False),
        patch("src.tmplot._helpers.tomotopy_installed", None),
        patch("src.tmplot._helpers.gensim_installed", None),
        patch("src.tmplot._helpers.bitermplus_installed", None),
    ):
        with pytest.warns(UserWarning, match="tomotopy, gensim, bitermplus"):
            with pytest.raises(ValueError, match="Unsupported model type"):
                tm.get_phi(object())


def test_gensim_theta_is_built_from_sparse_topic_probabilities():
    model = Mock(num_topics=2)
    model.get_document_topics.side_effect = [[(1, 0.8)], [(0, 0.6), (1, 0.4)]]
    with (
        patch("src.tmplot._helpers._is_tomotopy", return_value=False),
        patch("src.tmplot._helpers._is_gensim", return_value=True),
    ):
        result = tm.get_theta(model, corpus=["first", "second"])
    assert result.to_numpy().tolist() == [[0.0, 0.6], [0.8, 0.4]]


def test_entropy_handles_uniform_distribution_boundary():
    assert np.isfinite(tm.entropy(np.full((2, 2), 0.5)))


@pytest.mark.parametrize(
    "phi",
    [np.array([]), np.array([[np.nan]]), np.array([[-1.0]])],
)
def test_entropy_rejects_invalid_inputs(phi):
    with pytest.raises(ValueError):
        tm.entropy(phi)


def test_stability_aligns_vocabulary_labels():
    reference = pd.DataFrame({0: [0.9, 0.1]}, index=["a", "b"])
    reordered = pd.DataFrame({0: [0.1, 0.9]}, index=["b", "a"])
    with patch("src.tmplot._stability.get_phi", side_effect=[reference, reordered]):
        _, distances = tm.get_closest_topics([object(), object()], verbose=False)
    assert distances[0, 1] == pytest.approx(0)


def test_stability_supports_different_topic_counts():
    reference = pd.DataFrame({0: [0.9, 0.1], 1: [0.1, 0.9]}, index=["a", "b"])
    current = pd.DataFrame({0: [0.9, 0.1]}, index=["a", "b"])
    with patch("src.tmplot._stability.get_phi", side_effect=[reference, current]):
        closest, distances = tm.get_closest_topics([object(), object()], verbose=False)
    assert closest.shape == distances.shape == (2, 2)


def test_stability_renormalizes_shared_vocabulary():
    reference = pd.DataFrame({0: [0.45, 0.05, 0.5]}, index=["a", "b", "only_ref"])
    current = pd.DataFrame({0: [0.9, 0.1, 0.0]}, index=["a", "b", "only_current"])
    with patch("src.tmplot._stability.get_phi", side_effect=[reference, current]):
        _, distances = tm.get_closest_topics(
            [object(), object()], method="tv", verbose=False
        )
    assert distances[0, 1] == pytest.approx(0)


def test_identical_topics_remain_stable_after_normalization():
    closest = np.array([[0, 0, 0]])
    stable, distances = tm.get_stable_topics(closest, np.zeros((1, 3)), thres_models=2)
    assert stable.tolist() == [[0, 0, 0]]
    assert distances.tolist() == [[1.0, 1.0, 1.0]]


def test_plot_scatter_validates_optional_columns_after_ndarray_conversion():
    with pytest.raises(ValueError, match="size column"):
        tm.plot_scatter_topics(np.array([[0.0, 1.0]]), size_col="size")


def test_prepare_coords_rejects_wrong_label_count():
    with (
        patch("src.tmplot._report.get_phi", return_value=pd.DataFrame([[1.0, 1.0]])),
        patch(
            "src.tmplot._report.get_theta", return_value=pd.DataFrame([[1.0], [1.0]])
        ),
        patch("src.tmplot._report.get_topics_dist", return_value=np.zeros((2, 2))),
        patch(
            "src.tmplot._report.get_topics_scatter",
            return_value=pd.DataFrame({"x": [0, 1], "y": [0, 1]}),
        ),
    ):
        with pytest.raises(ValueError, match="labels length"):
            tm.prepare_coords(object(), labels=["only one"])


def test_report_skips_disabled_panel_computation():
    phi = pd.DataFrame({0: [1.0]}, index=["word"])
    with (
        patch("src.tmplot._report.get_phi", return_value=phi),
        patch("src.tmplot._report.get_theta") as get_theta,
        patch("src.tmplot._report.prepare_coords") as prepare_coords,
    ):
        tm.report(
            object(), ["document"], show_topics=False, show_words=False, show_docs=False
        )
    get_theta.assert_not_called()
    prepare_coords.assert_not_called()


def test_report_rejects_short_topic_labels():
    phi = pd.DataFrame({0: [0.5], 1: [0.5]}, index=["word"])
    with patch("src.tmplot._report.get_phi", return_value=phi):
        with pytest.raises(ValueError, match="topics_labels length"):
            tm.report(
                object(),
                ["document"],
                topics_labels=["one"],
                show_topics=False,
                show_words=False,
                show_docs=False,
            )


def test_report_preserves_custom_top_document_options():
    phi = pd.DataFrame({0: [0.5], 1: [0.5]}, index=["word"])
    theta = pd.DataFrame([[0.8], [0.2]])
    custom_docs = pd.DataFrame({"custom": ["document"]})
    with (
        patch("src.tmplot._report.get_phi", return_value=phi),
        patch("src.tmplot._report.get_theta", return_value=theta),
        patch("src.tmplot._report.get_top_docs") as get_top_docs,
    ):
        report = tm.report(
            object(),
            ["document"],
            show_topics=False,
            show_words=False,
            docs_kws={"docs": custom_docs},
            top_docs_kws={"docs_num": 1},
        )
        report.children[0].children[0].children[1].value = 1
    assert get_top_docs.call_args.kwargs["docs_num"] == 1
    assert get_top_docs.call_args.kwargs["topics"] == [1]


def test_report_caches_distances_when_embedding_method_changes():
    phi = pd.DataFrame({0: [0.5], 1: [0.5]}, index=["word"])
    theta = pd.DataFrame([[0.8], [0.2]])
    coords = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "topic": [0, 1],
            "size": [80, 20],
            "label": [0, 1],
        }
    )
    with (
        patch("src.tmplot._report.get_phi", return_value=phi),
        patch("src.tmplot._report.get_theta", return_value=theta),
        patch("src.tmplot._report.prepare_coords", return_value=coords),
        patch(
            "src.tmplot._report.get_topics_dist", return_value=np.zeros((2, 2))
        ) as get_dist,
        patch(
            "src.tmplot._report.get_topics_scatter",
            return_value=coords.drop(columns="label"),
        ),
    ):
        report = tm.report(object(), ["document"], show_words=False, show_docs=False)
        method_dropdown = report.children[2].children[0].children[1].children[1]
        method_dropdown.value = "mds"
        method_dropdown.value = "isomap"
    get_dist.assert_called_once()
