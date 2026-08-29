import json
import warnings
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from ipywidgets import widgets as wdg

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
    ("distances", "theta", "message"),
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
        patch("src.tmplot._helpers._is_btm_classifier", return_value=False),
        patch("src.tmplot._helpers.tomotopy_installed", True),
        patch("src.tmplot._helpers.gensim_installed", True),
        patch("src.tmplot._helpers.bitermplus_installed", True),
    ):
        with pytest.raises(ValueError, match="Unsupported model type"):
            tm.get_phi(object())


def test_optional_adapter_checks_do_not_warn_for_unrelated_objects():
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
    ("phi", "match"),
    [
        (np.array([]), "non-empty 2D"),
        (np.array([[np.nan]]), "finite non-negative"),
        (np.array([[-1.0]]), "finite non-negative"),
    ],
)
def test_entropy_rejects_invalid_inputs(phi, match):
    with pytest.raises(ValueError, match=match):
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
        ),pytest.raises(ValueError, match="labels length")
    ):
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


# --- Vectorized distance matrix must match the scalar pairwise reference ---

DISTANCE_METHODS = ["klb", "sklb", "jsd", "jef", "hel", "bhat", "tv", "jac"]


def _naive_topics_dist(phi, method, **kwargs):
    """Reference implementation: the original O(T^2) pairwise loop."""
    from itertools import combinations

    dist_func = getattr(tm._distance, f"_dist_{method}")
    topics_num = phi.shape[1]
    dists = np.zeros((topics_num, topics_num))
    for i, j in combinations(range(topics_num), 2):
        dists[((i, j), (j, i))] = dist_func(phi[:, i], phi[:, j], **kwargs)
    return dists


@pytest.mark.parametrize("method", DISTANCE_METHODS)
def test_vectorized_distances_match_pairwise_reference(method):
    phi = np.random.default_rng(42).random((200, 8))
    phi /= phi.sum(axis=0)
    np.testing.assert_allclose(
        tm.get_topics_dist(phi, method=method),
        _naive_topics_dist(phi, method),
        rtol=0,
        atol=1e-10,
    )


@pytest.mark.parametrize("top_words", [1, 3, 50, 500])
def test_vectorized_jaccard_matches_pairwise_reference(top_words):
    phi = np.random.default_rng(7).random((200, 8))
    phi /= phi.sum(axis=0)
    np.testing.assert_array_equal(
        tm.get_topics_dist(phi, method="jac", top_words=top_words),
        _naive_topics_dist(phi, "jac", top_words=top_words),
    )


@pytest.mark.parametrize("method", DISTANCE_METHODS)
def test_closest_topics_matches_pairwise_reference(method):
    rng = np.random.default_rng(3)
    words = [f"w{i}" for i in range(120)]
    frames = []
    for topics_num in (5, 4):
        values = rng.random((120, topics_num))
        frames.append(pd.DataFrame(values / values.sum(axis=0), index=words))
    reference, current = frames  # pylint: disable=unbalanced-tuple-unpacking

    dist_func = getattr(tm._distance, f"_dist_{method}")
    ref_values = reference.to_numpy()
    cur_values = current.to_numpy()
    expected = np.array(
        [
            [
                dist_func(ref_values[:, i], cur_values[:, j])
                for j in range(cur_values.shape[1])
            ]
            for i in range(ref_values.shape[1])
        ]
    )

    with patch("src.tmplot._stability.get_phi", side_effect=[reference, current]):
        closest, distances = tm.get_closest_topics(
            [object(), object()], method=method, verbose=False
        )

    np.testing.assert_array_equal(closest[:, 1], expected.argmin(axis=1))
    np.testing.assert_allclose(distances[:, 1], expected.min(axis=1), atol=1e-10)


# --- Correctness regressions ---


@pytest.mark.parametrize("top_words", [0, -5])
def test_jaccard_rejects_non_positive_top_words(top_words):
    with pytest.raises(ValueError, match="top_words"):
        tm._distance._dist_jac(
            np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5]), top_words=top_words
        )


@pytest.mark.parametrize("top_words", [0, -5])
def test_get_topics_dist_rejects_non_positive_top_words(top_words):
    phi = np.array([[0.7, 0.3], [0.3, 0.7]])
    with pytest.raises(ValueError, match="top_words"):
        tm.get_topics_dist(phi, method="jac", top_words=top_words)


def test_report_accepts_array_like_docs():
    phi = pd.DataFrame({0: [0.6, 0.4], 1: [0.3, 0.7]}, index=["a", "b"])
    theta = pd.DataFrame({0: [0.5, 0.5], 1: [0.2, 0.8]})
    with (
        patch("src.tmplot._report.get_phi", return_value=phi),
        patch("src.tmplot._report.get_theta", return_value=theta),
    ):
        report = tm.report(
            object(), np.array(["first doc", "second doc"]), show_topics=False
        )
    assert isinstance(report, wdg.VBox)


@pytest.mark.parametrize("docs", [[], np.array([]), None])
def test_report_still_rejects_empty_docs(docs):
    with pytest.raises(ValueError, match="docs cannot be empty"):
        tm.report(object(), docs)


def test_scatter_highlight_follows_custom_topic_col():
    coords = pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "tid": [0, 1], "size": [1.0, 1.0]}
    )
    chart = tm.plot_scatter_topics(coords, topic=1, topic_col="tid", size_col="size")
    spec = json.dumps(chart.to_dict())
    assert "datum['tid'] == 1" in spec
    assert "datum['topic']" not in spec


def test_terms_probs_ratio_handles_duplicate_word_labels():
    phi = pd.DataFrame(
        np.array([[0.5, 0.2], [0.3, 0.3], [0.2, 0.5]]), index=["a", "a", "b"]
    )
    result = tm.calc_terms_probs_ratio(phi, topic=0, terms_num=3)
    terms = result.loc[result["Type"].str.startswith("Conditional"), "Terms"]
    assert len(terms) == 3
    assert sorted(terms.tolist()) == ["a", "a", "b"]


@pytest.mark.parametrize("ref", [-1, 5])
def test_closest_topics_rejects_out_of_range_ref(ref):
    with pytest.raises(ValueError, match="ref"):
        tm.get_closest_topics([object(), object()], ref=ref, verbose=False)


def test_closest_topics_rejects_empty_models():
    with pytest.raises(ValueError, match="at least one model"):
        tm.get_closest_topics([], verbose=False)


def test_stable_topics_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        tm.get_stable_topics(np.zeros((2, 3), dtype=int), np.zeros((2, 4)))


def test_stable_topics_rejects_out_of_range_ref():
    with pytest.raises(ValueError, match="ref"):
        tm.get_stable_topics(np.zeros((2, 3), dtype=int), np.zeros((2, 3)), ref=7)


# --- get_top_docs bounds checking ---


@pytest.mark.parametrize("topics", [[9], [0, 5], [-1]])
def test_get_top_docs_rejects_out_of_range_topics(topics):
    theta = np.array([[0.9, 0.1], [0.2, 0.8]])
    with pytest.raises(IndexError, match=r"topics contains indices outside \[0, 1\]"):
        tm.get_top_docs(["a", "b"], theta=theta, topics=topics)


def test_get_top_docs_accepts_in_range_topics():
    theta = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = tm.get_top_docs(["a", "b"], theta=theta, topics=[1])
    assert list(result.columns) == ["topic1"]


# --- Dead topics of nonparametric (HDP) models ---


class _ParametricModel:
    """Stands in for LDA/CTM/DMR/...: a fixed topic count, no dead topics."""

    def __init__(self, k):
        self.k = k


class _NonparametricModel(_ParametricModel):
    """Stands in for HDPModel, where some of the ``k`` topics have died out."""

    def __init__(self, k, live):
        super().__init__(k)
        self._live = set(live)

    def is_live_topic(self, topic):
        return topic in self._live


def test_live_topic_ids_passes_through_parametric_models():
    # No is_live_topic attribute: every topic is real (LDA, CTM, DMR, ...).
    assert tm._helpers._live_topic_ids(_ParametricModel(4)) == [0, 1, 2, 3]


def test_live_topic_ids_drops_dead_topics_and_warns():
    model = _NonparametricModel(6, live=[1, 4])
    with pytest.warns(UserWarning, match="Dropped 4 dead topic"):
        assert tm._helpers._live_topic_ids(model) == [1, 4]


def test_live_topic_ids_is_silent_when_all_topics_are_live():
    model = _NonparametricModel(3, live=[0, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tm._helpers._live_topic_ids(model) == [0, 1, 2]


def test_manifest_only_references_existing_files():
    root = Path(__file__).resolve().parent.parent
    missing = [
        line.split(None, 1)[1].strip()
        for line in (root / "MANIFEST.in").read_text().splitlines()
        if line.startswith("include ")
        and not (root / line.split(None, 1)[1].strip()).exists()
    ]
    assert missing == [], f"MANIFEST.in references missing files: {missing}"


# --- entropy orientation ---


def _make_phi(words, topics, seed=0):
    """W x T phi: each column is a topic's distribution over words."""
    phi = np.random.default_rng(seed).random((words, topics))
    return phi / phi.sum(axis=0)


@pytest.mark.parametrize("shape", [(40, 30), (60, 50), (25, 20), (100, 5)])
def test_entropy_is_orientation_independent(shape):
    # Passing phi either way must give the same answer; previously the W x T
    # form silently returned a different, meaningless number.
    phi = _make_phi(*shape)
    assert tm.entropy(phi) == pytest.approx(tm.entropy(phi.T))


def test_entropy_matches_the_documented_t_by_w_result():
    # Backward compatibility: correct T x W callers are unaffected.
    phi = _make_phi(40, 30)
    assert tm.entropy(phi.T) == pytest.approx(tm.entropy(phi.T, topics_axis=0))


@pytest.mark.parametrize("max_probs", [False, True])
def test_entropy_explicit_axis_agrees_with_inference(max_probs):
    phi = _make_phi(40, 30)
    assert tm.entropy(phi, topics_axis=1, max_probs=max_probs) == pytest.approx(
        tm.entropy(phi.T, topics_axis=0, max_probs=max_probs)
    )


def test_entropy_rejects_topics_axis_contradicted_by_phi():
    phi = _make_phi(40, 30)  # W x T, so topics_axis=1
    with pytest.raises(ValueError, match="declares a T x W matrix"):
        tm.entropy(phi, topics_axis=0)
    with pytest.raises(ValueError, match="declares a W x T matrix"):
        tm.entropy(phi.T, topics_axis=1)


def test_entropy_warns_when_orientation_cannot_be_verified():
    unnormalized = np.full((4, 10), 0.3)
    with pytest.warns(UserWarning, match="Neither axis of phi sums to 1"):
        tm.entropy(unnormalized)
    with pytest.warns(UserWarning, match="Neither axis of phi sums to 1"):
        tm.entropy(unnormalized, topics_axis=0)


@pytest.mark.parametrize("topics_axis", [2, -1, "rows"])
def test_entropy_rejects_invalid_topics_axis(topics_axis):
    with pytest.raises(ValueError, match="topics_axis must be 0 or 1"):
        tm.entropy(_make_phi(10, 4), topics_axis=topics_axis)


def test_entropy_square_doubly_stochastic_is_accepted():
    # Orientation invariant for a square matrix unless max_probs picks an axis.
    assert np.isfinite(tm.entropy(np.full((2, 2), 0.5)))
    assert np.isfinite(tm.entropy(np.full((2, 2), 0.5), max_probs=True))


def test_entropy_square_ambiguous_under_max_probs_requires_explicit_axis():
    # A 2x2 doubly stochastic matrix is always symmetric, so use a 3x3
    # circulant one: every row and every column sums to 1, but phi != phi.T.
    phi = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
    assert np.allclose(phi.sum(0), 1)
    assert np.allclose(phi.sum(1), 1)
    assert not np.allclose(phi, phi.T)
    with pytest.raises(ValueError, match="cannot infer the topics axis"):
        tm.entropy(phi, max_probs=True)
    assert np.isfinite(tm.entropy(phi, max_probs=True, topics_axis=0))


# =====================================================================
# Tier 1: defects found by probing, fixed alongside these tests
# =====================================================================


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("method", ["tsne", "sem", "mds", "lle", "ltsa", "isomap"])
def test_scatter_survives_a_degenerate_distance_matrix(method):
    """D1: every topic identical, so all pairwise distances collapse to zero.

    Classical MDS then yields fewer dimensions than the two the embedding is
    asked for, which used to make lle and ltsa raise.
    """
    coords = tm.get_topics_scatter(np.zeros((4, 4)), np.ones((4, 2)), method=method)
    assert coords[["x", "y"]].shape == (4, 2)
    assert np.isfinite(coords[["x", "y"]].to_numpy()).all()


def test_classical_mds_pads_to_the_requested_dimensions():
    coords = tm._distance._classical_mds(np.zeros((5, 5)), n_components=2)
    assert coords.shape == (5, 2)
    assert np.isfinite(coords).all()


def test_hdp_with_no_live_topics_fails_loudly():
    """D2: dropping every topic would leave phi with zero columns."""
    model = _NonparametricModel(5, live=[])
    with pytest.raises(ValueError, match="no live topics"):
        tm._helpers._live_topic_ids(model)


# =====================================================================
# Tier 4: numerical edge cases that line coverage cannot find
# =====================================================================


@pytest.mark.parametrize("method", DISTANCE_METHODS)
def test_identical_topics_are_at_zero_distance(method):
    """N1: the one property all eight measures must share."""
    phi = np.repeat(_make_phi(60, 1), 3, axis=1)
    assert np.allclose(tm.get_topics_dist(phi, method=method), 0.0, atol=1e-9)


def test_disjoint_topics_have_bounded_and_stable_distances():
    """N2: with no shared support the divergences rest on the EPSILON floor.

    The bounded measures have exact analytic values. The unbounded ones are
    pinned so that changing EPSILON, which would silently rescale every
    get_stable_topics threshold, fails here instead of passing quietly.
    """
    phi = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert tm.get_topics_dist(phi, method="hel")[0, 1] == pytest.approx(1.0)
    assert tm.get_topics_dist(phi, method="tv")[0, 1] == pytest.approx(1.0)
    assert tm.get_topics_dist(phi, method="jsd")[0, 1] == pytest.approx(np.log(2))
    sklb = tm.get_topics_dist(phi, method="sklb")[0, 1]
    bhat = tm.get_topics_dist(phi, method="bhat")[0, 1]
    assert sklb == pytest.approx(294.73, abs=0.01)
    assert bhat == pytest.approx(72.99, abs=0.01)


def _phi_with_ties(words=80, topics=5, seed=0):
    """A phi whose probability plateau straddles the top-k cutoff.

    Which of the tied words land in the top k is decided purely by the
    selection algorithm, so argsort and argpartition disagree here. A phi of
    uniformly equal values would NOT expose that: every column is then
    identical and both algorithms return the same set.
    """
    rng = np.random.default_rng(seed)
    column = np.concatenate(
        [np.full(3, 0.9), np.full(40, 0.5), np.full(words - 43, 0.1)]
    )
    phi = np.column_stack([rng.permutation(column) for _ in range(topics)])
    return phi / phi.sum(axis=0)


@pytest.mark.parametrize("top_words", [1, 10, 25, 80])
def test_jaccard_tie_breaking_matches_the_pairwise_reference(top_words):
    """N3: the vectorized Jaccard keeps argsort, not the faster argpartition,
    precisely so ties resolve the same way as the scalar reference."""
    phi = _phi_with_ties()
    np.testing.assert_array_equal(
        tm.get_topics_dist(phi, method="jac", top_words=top_words),
        _naive_topics_dist(phi, "jac", top_words=top_words),
    )


def test_jaccard_ties_are_actually_exercised_by_the_fixture():
    """Guards the guard: if the fixture stopped producing ties across the
    cutoff, the test above would pass no matter how top-k was selected.

    Asserting that argsort and argpartition happen to disagree would pin an
    unspecified numpy tie order. Pin instead the two properties that make
    top-k genuinely ambiguous: more words sit on the cutoff value than there
    are slots for them, and those tied words differ in whether the other
    topic also selected them, so the choice moves the Jaccard overlap.
    """
    phi = _phi_with_ties()
    top_words = 10

    def cutoff_ties(topic):
        column = phi[:, topic]
        cutoff = np.sort(column)[-top_words]
        selected = set(np.argsort(column)[-top_words:].tolist())
        on_cutoff = set(np.flatnonzero(column == cutoff).tolist())
        return selected, on_cutoff

    first, first_ties = cutoff_ties(0)
    second, _ = cutoff_ties(1)
    assert len(first_ties) > len(first_ties & first)
    assert {word in second for word in first_ties} == {True, False}


def test_single_topic_behaviour_across_the_api():
    """N4: pins what each entry point currently does for T = 1."""
    phi = np.array([[0.6], [0.4]])
    assert tm.get_topics_dist(phi).shape == (1, 1)
    assert np.allclose(tm.get_topics_dist(phi), 0.0)

    with pytest.raises(ValueError, match="at least two topics"):
        tm.get_topics_scatter(np.zeros((1, 1)), np.ones((1, 3)))

    # Renyi entropy is undefined at T = 1: the deformation parameter q = 1/T
    # equals 1, so the F/(q - 1) denominator vanishes. Raising here matches
    # bitermplus 1.0.
    with pytest.raises(ValueError, match="undefined for a single topic"):
        tm.entropy(np.full((1, 50), 0.02))

    ratio = tm.calc_terms_probs_ratio(pd.DataFrame(phi), topic=0)
    assert set(ratio["Type"]) == {
        "Conditional term probability, p(w | t)",
        "Marginal term probability, p(w)",
    }


def test_a_topic_no_document_uses_gets_zero_size():
    """N5: a topic with no document mass must not distort the scatter plot."""
    theta = np.array([[1.0, 1.0], [0.0, 0.0]])
    coords = tm.get_topics_scatter(np.array([[0.0, 1.0], [1.0, 0.0]]), theta)
    assert coords["size"].tolist() == [100.0, 0.0]


def test_a_document_with_no_topic_mass_is_tolerated():
    """N5 (twin): an empty document column must not break normalization."""
    theta = np.array([[0.5, 0.0], [0.5, 0.0]])
    assert tm.calc_topics_marg_probs(theta).tolist() == [0.5, 0.5]


def test_normalisation_tolerance_boundary():
    """N6: get_topics_dist accepts columns summing to 1 only within tolerance."""
    phi = _make_phi(50, 3)
    phi[0, 0] += 9e-7
    tm.get_topics_dist(phi)  # inside tolerance, must not raise

    phi[0, 0] += 1e-3
    with pytest.raises(ValueError, match="sum to 1"):
        tm.get_topics_dist(phi)


@pytest.mark.parametrize("method", DISTANCE_METHODS)
def test_large_vocabulary_does_not_underflow(method):
    """N8: real vocabularies push probabilities far below any small fixture."""
    phi = _make_phi(20000, 6, seed=5)
    assert phi.min() < 1e-6
    dists = tm.get_topics_dist(phi, method=method)
    assert np.isfinite(dists).all()
    assert (dists >= 0).all()
    assert np.allclose(np.diag(dists), 0.0)


# =====================================================================
# Tier 3: validation contracts, pinning error type and message
# =====================================================================


@pytest.mark.parametrize(
    ("phi", "message"),
    [
        (np.ones(3), "2D array"),
        (np.array([[[0.5]]]), "2D array"),
        (np.array([[0.5, -0.5], [0.5, 1.5]]), "non-negative"),
        (np.array([[0.5, 0.2], [0.2, 0.5]]), "sum to 1"),
    ],
)
def test_get_topics_dist_rejects_a_malformed_phi(phi, message):
    """V1"""
    with pytest.raises(ValueError, match=message):
        tm.get_topics_dist(phi)


@pytest.mark.parametrize("topics_num", [0, 1])
def test_get_topics_scatter_needs_at_least_two_topics(topics_num):
    """V2"""
    with pytest.raises(ValueError, match="at least two topics"):
        tm.get_topics_scatter(
            np.zeros((topics_num, topics_num)), np.ones((topics_num, 2))
        )


@pytest.mark.parametrize("method", [m for m in DISTANCE_METHODS if m != "jac"])
def test_cross_dists_rejects_arguments_a_measure_cannot_use(method):
    """V3: top_words means nothing outside Jaccard, so saying so beats ignoring it."""
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        tm._distance._cross_dists(np.eye(3), np.eye(3), method, top_words=5)


def test_registering_a_distance_without_implementing_it_is_caught():
    """V3: guards a maintainer adding a name to the registry and nothing else."""
    registry = dict(tm._distance.DIST_FUNCS, brand_new=lambda a, b: 0.0)
    with patch.object(tm._distance, "DIST_FUNCS", registry), pytest.raises(AssertionError, match="was not handled"):
        tm._distance._cross_dists(np.eye(3), np.eye(3), "brand_new")


def test_registering_a_scatter_method_without_implementing_it_is_caught():
    """V3 (twin) for get_topics_scatter."""
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    with patch.object(
        tm._distance, "SCATTER_METHODS", [*tm._distance.SCATTER_METHODS, "brand_new"]
    ), pytest.raises(AssertionError, match="was not handled"):
        tm.get_topics_scatter(distances, np.ones((3, 2)), method="brand_new")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"theta": np.ones(3)}, ValueError, "2D topics x documents"),
        ({"theta": np.ones((2, 2)), "docs_num": 0}, ValueError, "docs_num must be"),
        ({"theta": np.ones((2, 2)), "docs_num": -1}, ValueError, "docs_num must be"),
    ],
)
def test_get_top_docs_validates_its_inputs(kwargs, error, message):
    """V4"""
    with pytest.raises(error, match=message):
        tm.get_top_docs(["a", "b"], **kwargs)


@pytest.mark.parametrize(
    ("phi", "p_t", "message"),
    [
        (np.ones(4), np.ones(2), "phi matrix must be a 2D"),
        (np.ones((2, 2)), np.ones((2, 2)), "p_t array must be a 1D"),
        (np.ones((2, 2)), np.array([-1.0, 2.0]), "p_t must contain finite"),
        (np.ones((2, 2)), np.array([np.nan, 1.0]), "p_t must contain finite"),
    ],
)
def test_calc_terms_marg_probs_validates_its_inputs(phi, p_t, message):
    """V5"""
    with pytest.raises(ValueError, match=message):
        tm.calc_terms_marg_probs(phi, p_t)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"phi": np.ones(4), "topic": 0}, ValueError, "2D words x topics"),
        ({"phi": np.ones((3, 2)), "topic": 5}, IndexError, "out of bounds"),
        ({"phi": np.ones((3, 2)), "topic": -1}, IndexError, "out of bounds"),
        (
            {"phi": np.ones((3, 2)), "topic": 0, "p_t": np.ones(5)},
            ValueError,
            "p_t length must match",
        ),
        (
            {"phi": np.ones((3, 2)), "topic": 0, "lambda_": 1.5},
            ValueError,
            "lambda_ must be between",
        ),
        (
            {"phi": np.ones((3, 2)), "topic": 0, "lambda_": -0.1},
            ValueError,
            "lambda_ must be between",
        ),
    ],
)
def test_get_relevant_terms_validates_its_inputs(kwargs, error, message):
    """V6"""
    with pytest.raises(error, match=message):
        tm.get_relevant_terms(**kwargs)


def test_closest_topics_rejects_models_with_no_shared_vocabulary():
    """V7: comparing models built from different corpora."""
    first = pd.DataFrame({0: [0.5, 0.5]}, index=["a", "b"])
    second = pd.DataFrame({0: [0.5, 0.5]}, index=["c", "d"])
    with patch("src.tmplot._stability.get_phi", side_effect=[first, second]):
        with pytest.raises(ValueError, match="any vocabulary terms in common"):
            tm.get_closest_topics([object(), object()], verbose=False)


def test_closest_topics_rejects_a_topic_with_no_shared_probability_mass():
    """V7: the vocabularies overlap, but a topic puts nothing on the overlap."""
    first = pd.DataFrame({0: [0.0, 0.0, 1.0]}, index=["a", "b", "only_first"])
    second = pd.DataFrame({0: [0.5, 0.5, 0.0]}, index=["a", "b", "only_second"])
    with patch("src.tmplot._stability.get_phi", side_effect=[first, second]):
        with pytest.raises(ValueError, match="zero probability mass"):
            tm.get_closest_topics([object(), object()], verbose=False)


@pytest.mark.parametrize(
    ("closest", "dist"),
    [(np.zeros(3, dtype=int), np.zeros(3)), (np.zeros((2, 2), dtype=int), np.zeros(2))],
)
def test_stable_topics_rejects_non_2d_input(closest, dist):
    """V8"""
    with pytest.raises(ValueError, match="both be 2D arrays"):
        tm.get_stable_topics(closest, dist)


def test_plot_scatter_reports_a_missing_label_column():
    """V9"""
    coords = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="label column"):
        tm.plot_scatter_topics(coords, label_col="label")


@pytest.mark.parametrize("phi", [np.full((1, 50), 0.02), np.full((50, 1), 0.02)])
def test_entropy_rejects_a_single_topic_in_either_orientation(phi):
    """The check must sit after orientation is resolved, not before."""
    with pytest.raises(ValueError, match="undefined for a single topic"):
        tm.entropy(phi)


def test_entropy_matches_bitermplus_for_the_shared_default():
    """tmplot's max_probs=False and bitermplus's max_probs=True are the same
    computation; the two packages just name the flag in opposite ways."""
    bitermplus = pytest.importorskip("bitermplus")
    rng = np.random.default_rng(7)
    for topics, words in [(5, 200), (12, 800), (3, 40)]:
        p_wz = rng.random((topics, words))
        p_wz /= p_wz.sum(axis=1, keepdims=True)
        assert tm.entropy(p_wz) == pytest.approx(bitermplus.entropy(p_wz))


def test_entropy_rejects_a_phi_with_nothing_above_the_threshold():
    """V10.

    Only reachable for an unnormalized phi. When each topic sums to 1 across W
    words the largest entry is at least the mean 1/W, and the threshold test
    uses >=, so a valid phi always has something at or above it.
    """
    normalized = np.full((3, 40), 1 / 40)
    assert np.isfinite(tm.entropy(normalized, topics_axis=0))

    unnormalized = np.full((3, 10), 1 / 300)
    with pytest.warns(UserWarning, match="Neither axis of phi sums to 1"):
        with pytest.raises(ValueError, match="at or above the threshold"):
            tm.entropy(unnormalized, topics_axis=0)


# =====================================================================
# Tier 5: behaviour that works today and should keep working
# =====================================================================


def test_document_text_is_escaped():
    """L1: documents are user data rendered as HTML inside a notebook.

    Escaping holds because DataFrame.to_html escapes by default, which is one
    html_kws={"escape": False} away from being switched off.
    """
    html = tm.plot_docs(["<script>alert(1)</script>", "a & b < c"]).data
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html


def test_escaping_survives_a_dataframe_of_documents():
    """L1: the DataFrame branch of plot_docs takes a different path."""
    docs = pd.DataFrame({"": ["<img src=x onerror=alert(1)>"]})
    html = tm.plot_docs(docs).data
    assert "<img" not in html
    assert "&lt;img" in html


def test_duplicate_word_labels_do_not_corrupt_relevance():
    """L2: the neighbours of the function that was fixed for duplicates."""
    phi = pd.DataFrame(
        np.array([[0.5, 0.2], [0.3, 0.3], [0.2, 0.5]]), index=["a", "a", "b"]
    )
    relevance = tm.get_relevant_terms(phi, topic=0)
    assert len(relevance) == 3
    assert sorted(relevance.index) == ["a", "a", "b"]
    assert relevance.is_monotonic_decreasing

    top_words = tm.get_top_topic_words(phi, words_num=2)
    assert top_words.shape == (2, 2)


@pytest.mark.parametrize(
    ("closest", "dist", "kwargs", "expected_rows"),
    [
        (np.zeros((2, 1), dtype=int), np.zeros((2, 1)), {}, 0),
        (np.zeros((2, 3), dtype=int), np.zeros((2, 3)), {"thres_models": 9}, 0),
    ],
)
def test_degenerate_stability_inputs_return_empty(closest, dist, kwargs, expected_rows):
    """L3: no stable topics is a valid answer, and must not be garbage."""
    stable, distances = tm.get_stable_topics(closest, dist, **kwargs)
    assert stable.shape == (expected_rows, closest.shape[1])
    assert distances.shape == (expected_rows, closest.shape[1])


# =====================================================================
# Tier 2: the interactive callbacks of report()
# =====================================================================


def _report_matrices():
    phi = pd.DataFrame(
        {0: [0.5, 0.3, 0.2], 1: [0.2, 0.5, 0.3], 2: [0.3, 0.2, 0.5]},
        index=["alpha", "beta", "gamma"],
    )
    theta = pd.DataFrame(
        np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    )
    return phi, theta


def _build_report(**kwargs):
    """A report over a 3-word, 3-topic, 3-document model."""
    phi, theta = _report_matrices()
    with (
        patch("src.tmplot._report.get_phi", return_value=phi),
        patch("src.tmplot._report.get_theta", return_value=theta),
    ):
        return tm.report(object(), ["doc one", "doc two", "doc three"], **kwargs)


def _widgets_of(root, kind):
    found, stack = [], [root]
    while stack:
        node = stack.pop()
        if isinstance(node, kind):
            found.append(node)
        stack.extend(getattr(node, "children", ()))
    return found


def _topic_dropdown(report):
    # The topic selector holds an int; the embedding-method one holds a string.
    return next(
        d for d in _widgets_of(report, wdg.Dropdown) if not isinstance(d.value, str)
    )


def test_selecting_a_topic_redraws_every_panel():
    """R1: the one callback that touches all three panels."""
    report = _build_report()
    with (
        patch("src.tmplot._report.display"),
        patch("src.tmplot._report.plot_terms") as plot_terms,
        patch("src.tmplot._report.plot_scatter_topics") as plot_scatter,
        patch("src.tmplot._report.plot_docs") as plot_docs,
        patch("src.tmplot._report.calc_terms_probs_ratio") as calc_terms,
        patch("src.tmplot._report.get_top_docs") as get_top_docs,
    ):
        calc_terms.return_value = pd.DataFrame(
            {"Terms": ["alpha"], "Type": ["t"], "Probability": [1.0]}
        )
        get_top_docs.return_value = pd.DataFrame({"topic2": ["doc one"]})
        _topic_dropdown(report).value = 2

    assert plot_terms.called
    assert plot_scatter.called
    assert plot_docs.called
    # every panel followed the selection to the same topic
    assert calc_terms.call_args.kwargs["topic"] == 2
    assert plot_scatter.call_args.kwargs["topic"] == 2
    assert get_top_docs.call_args.kwargs["topics"] == [2]


def test_topic_callback_runs_with_only_one_panel_enabled():
    """R1: each branch is guarded separately, so a lone panel must still work."""
    for panels in (
        {"show_words": False, "show_docs": False},
        {"show_topics": False, "show_docs": False},
        {"show_topics": False, "show_words": False},
    ):
        report = _build_report(**panels)
        with (
            patch("src.tmplot._report.display"),
            patch("src.tmplot._report.plot_terms"),
            patch("src.tmplot._report.plot_scatter_topics"),
            patch("src.tmplot._report.plot_docs"),
        ):
            _topic_dropdown(report).value = 1  # must not raise


def test_lambda_slider_uses_the_selected_topic():
    """R2: the slider must follow the dropdown, not fall back to topic 0."""
    report = _build_report(show_topics=False, show_docs=False)
    with (
        patch("src.tmplot._report.display"),
        patch("src.tmplot._report.plot_terms"),
        patch("src.tmplot._report.calc_terms_probs_ratio") as calc_terms,
    ):
        calc_terms.return_value = pd.DataFrame(
            {"Terms": ["alpha"], "Type": ["t"], "Probability": [1.0]}
        )
        _topic_dropdown(report).value = 1
        _widgets_of(report, wdg.FloatSlider)[0].value = 0.25

    assert calc_terms.call_args.kwargs["topic"] == 1
    assert calc_terms.call_args.kwargs["lambda_"] == pytest.approx(0.25)


def test_docs_slider_requests_that_many_documents():
    """R3"""
    report = _build_report(show_topics=False, show_words=False)
    with (
        patch("src.tmplot._report.display"),
        patch("src.tmplot._report.plot_docs"),
        patch("src.tmplot._report.get_top_docs") as get_top_docs,
    ):
        get_top_docs.return_value = pd.DataFrame({"topic0": ["doc one"]})
        _widgets_of(report, wdg.IntSlider)[0].value = 3

    assert get_top_docs.call_args.kwargs["docs_num"] == 3


def test_docs_slider_ceiling_respects_a_large_docs_num():
    """R3: a caller asking for more than 100 must not be clamped below it."""
    report = _build_report(show_topics=False, show_words=False,
                           top_docs_kws={"docs_num": 250})
    slider = _widgets_of(report, wdg.IntSlider)[0]
    assert slider.value == 250
    assert slider.max >= 250


# =====================================================================
# Tier 6: branches that depend on the installed environment
# =====================================================================


@contextmanager
def _no_model_packages():
    with (
        patch.object(tm._helpers, "tomotopy_installed", None),
        patch.object(tm._helpers, "gensim_installed", None),
        patch.object(tm._helpers, "bitermplus_installed", None),
    ):
        yield


def test_model_checks_are_false_without_their_packages():
    """E1: what a user who skipped the `models` extra sees."""
    with _no_model_packages():
        assert tm._helpers._is_tomotopy(object()) is False
        assert tm._helpers._is_gensim(object()) is False
        assert tm._helpers._is_btmplus(object()) is False
        assert tm._helpers._is_btm_classifier(object()) is False


@pytest.mark.parametrize("function", ["get_phi", "get_theta"])
def test_unsupported_model_names_the_missing_packages(function):
    """E1: the error should say what to install, not just that it failed."""
    with _no_model_packages(), pytest.warns(UserWarning, match="tomotopy, gensim, bitermplus"):
        with pytest.raises(ValueError, match="Unsupported model type"):
            getattr(tm, function)(object())


def test_mds_uses_the_legacy_argument_on_older_scikit_learn():
    """E2: pyproject allows scikit-learn>=1.4, so this branch is live for users.

    Before 1.9 the precomputed-distance argument was `dissimilarity`; from 1.9
    it is `metric`, and `init` became a constructor argument.
    """
    legacy_params = {
        "n_components": None,
        "dissimilarity": None,
        "normalized_stress": None,
        "n_init": None,
    }
    transformer = Mock()
    transformer.fit_transform.return_value = np.zeros((3, 2))
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    with (
        patch(
            "src.tmplot._distance.signature",
            return_value=Mock(parameters=legacy_params),
        ),
        patch("src.tmplot._distance.MDS", return_value=transformer) as constructor,
    ):
        tm.get_topics_scatter(distances, np.ones((3, 2)), method="mds")

    assert constructor.call_args.kwargs["dissimilarity"] == "precomputed"
    assert "metric" not in constructor.call_args.kwargs
    assert "init" not in constructor.call_args.kwargs


def test_mds_uses_the_modern_argument_on_current_scikit_learn():
    """E2: from scikit-learn 1.9 the branch flips to `metric` plus `init`."""
    modern_params = {
        "n_components": None,
        "metric_mds": None,
        "metric": None,
        "dissimilarity": None,
        "normalized_stress": None,
        "n_init": None,
        "init": None,
    }
    transformer = Mock()
    transformer.fit_transform.return_value = np.zeros((3, 2))
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    with (
        patch(
            "src.tmplot._distance.signature",
            return_value=Mock(parameters=modern_params),
        ),
        patch("src.tmplot._distance.MDS", return_value=transformer) as constructor,
    ):
        tm.get_topics_scatter(distances, np.ones((3, 2)), method="mds")

    assert constructor.call_args.kwargs["metric"] == "precomputed"
    assert constructor.call_args.kwargs["init"] == "random"
    assert "dissimilarity" not in constructor.call_args.kwargs


def test_mds_branch_matches_the_installed_scikit_learn():
    """E2: guards the detection itself against the real MDS signature."""
    from sklearn.manifold import MDS as InstalledMDS

    parameters = signature(InstalledMDS.__init__).parameters
    transformer = Mock()
    transformer.fit_transform.return_value = np.zeros((3, 2))
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    with (
        patch(
            "src.tmplot._distance.signature",
            return_value=Mock(parameters=parameters),
        ),
        patch("src.tmplot._distance.MDS", return_value=transformer) as constructor,
    ):
        tm.get_topics_scatter(distances, np.ones((3, 2)), method="mds")

    used = constructor.call_args.kwargs
    expected = "metric" if "metric_mds" in parameters else "dissimilarity"
    assert used[expected] == "precomputed"


# --- last uncovered branches ---


@pytest.mark.parametrize("lambda_", [-0.1, 1.5])
def test_terms_probs_ratio_validates_lambda(lambda_):
    """calc_terms_probs_ratio guards lambda_ separately from get_relevant_terms."""
    phi = pd.DataFrame({0: [0.6, 0.4], 1: [0.3, 0.7]}, index=["a", "b"])
    with pytest.raises(ValueError, match="lambda_ must be between"):
        tm.calc_terms_probs_ratio(phi, topic=0, lambda_=lambda_)


def test_closest_topics_rejects_an_unknown_distance_method():
    with pytest.raises(ValueError, match="Unknown distance method"):
        tm.get_closest_topics([object()], method="nope", verbose=False)


def test_plot_terms_defaults_every_optional_argument():
    """plot_terms must work with only its data, filling in all chart options."""
    terms_probs = pd.DataFrame(
        {
            "Terms": ["alpha", "beta"],
            "Type": ["Conditional", "Marginal"],
            "Probability": [0.6, 0.4],
        }
    )
    assert tm.plot_terms(terms_probs) is not None


def test_report_merges_caller_supplied_coords_kws():
    """coords_kws must reach prepare_coords, not be dropped."""
    phi, theta = _report_matrices()
    with (
        patch("src.tmplot._report.get_phi", return_value=phi),
        patch("src.tmplot._report.get_theta", return_value=theta),
        patch("src.tmplot._report.prepare_coords") as prepare_coords,
        patch("src.tmplot._report.plot_scatter_topics"),
    ):
        prepare_coords.return_value = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0],
                "topic": [0, 1, 2],
                "size": [40.0, 30.0, 30.0],
                "label": [0, 1, 2],
            }
        )
        tm.report(
            object(),
            ["a", "b", "c"],
            show_words=False,
            show_docs=False,
            coords_kws={"dist_kws": {"method": "hel"}, "labels": ["x", "y", "z"]},
        )
    assert prepare_coords.call_args.kwargs["dist_kws"] == {"method": "hel"}
    assert prepare_coords.call_args.kwargs["labels"] == ["x", "y", "z"]
