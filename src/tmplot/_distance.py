__all__ = ["get_topics_dist", "get_topics_scatter", "get_top_topic_words"]
from typing import Optional, Union, List
from inspect import signature
from pandas import DataFrame, Index
import numpy as np
from scipy.special import kl_div, xlogy
from scipy.spatial import distance
from sklearn.manifold import (
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
)
from ._helpers import calc_topics_marg_probs


EPSILON = 1e-64


SCATTER_METHODS = ["tsne", "sem", "mds", "lle", "ltsa", "isomap"]


def _validate_top_words(top_words: int) -> None:
    if not isinstance(top_words, (int, np.integer)) or top_words < 1:
        raise ValueError(f"top_words must be a positive integer, got {top_words!r}")


def _positive_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), EPSILON, None)
    return values / values.sum()


def _dist_klb(a1: np.ndarray, a2: np.ndarray):
    return kl_div(_positive_probabilities(a1), _positive_probabilities(a2)).sum()


def _dist_sklb(a1: np.ndarray, a2: np.ndarray):
    a1_safe = _positive_probabilities(a1)
    a2_safe = _positive_probabilities(a2)
    return (kl_div(a1_safe, a2_safe) + kl_div(a2_safe, a1_safe)).sum()


def _dist_jsd(a1: np.ndarray, a2: np.ndarray):
    m = 0.5 * (a1 + a2)
    dist = 0.5 * kl_div(a1, m) + 0.5 * kl_div(a2, m)
    return dist[np.isfinite(dist)].sum()


def _dist_jef(a1: np.ndarray, a2: np.ndarray):
    a1_safe = _positive_probabilities(a1)
    a2_safe = _positive_probabilities(a2)
    return ((a1_safe - a2_safe) * (np.log(a1_safe) - np.log(a2_safe))).sum()


def _dist_hel(a1: np.ndarray, a2: np.ndarray):
    a1_safe = a1.copy()
    a2_safe = a2.copy()
    a1_safe[(a1_safe <= 0) | ~np.isfinite(a1_safe)] = EPSILON
    a2_safe[(a2_safe <= 0) | ~np.isfinite(a2_safe)] = EPSILON
    hel_val = distance.euclidean(np.sqrt(a1_safe), np.sqrt(a2_safe)) / np.sqrt(2)
    return hel_val


def _dist_bhat(a1: np.ndarray, a2: np.ndarray):
    pq = a1 * a2
    pq[(pq <= 0) | ~np.isfinite(pq)] = EPSILON
    dist = -np.log(np.sum(np.sqrt(pq)))
    return dist


def _dist_tv(a1: np.ndarray, a2: np.ndarray):
    dist = np.sum(np.abs(a1 - a2)) / 2
    return dist


def _dist_jac(a1: np.ndarray, a2: np.ndarray, top_words=100):
    _validate_top_words(top_words)
    a = np.argsort(a1)[: -top_words - 1 : -1]
    b = np.argsort(a2)[: -top_words - 1 : -1]
    j_num = np.intersect1d(a, b, assume_unique=False).size
    j_den = np.union1d(a, b).size
    jac_val = 1 - j_num / j_den
    return jac_val


DIST_FUNCS = {
    "klb": _dist_klb,
    "sklb": _dist_sklb,
    "jsd": _dist_jsd,
    "jef": _dist_jef,
    "hel": _dist_hel,
    "bhat": _dist_bhat,
    "tv": _dist_tv,
    "jac": _dist_jac,
}


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    """Column-wise equivalent of :func:`_positive_probabilities`."""
    values = np.clip(np.asarray(values, dtype=float), EPSILON, None)
    return values / values.sum(axis=0, keepdims=True)


def _sanitize_columns(values: np.ndarray) -> np.ndarray:
    """Replace non-positive and non-finite entries with ``EPSILON``."""
    values = np.array(values, dtype=float)
    values[(values <= 0) | ~np.isfinite(values)] = EPSILON
    return values


def _cross_klb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """KL divergence of every column of ``a`` from every column of ``b``."""
    p_a = _normalize_columns(a)
    p_b = _normalize_columns(b)
    # KL(p || q) = sum_w p log p - sum_w p log q; the -p + q terms of ``kl_div``
    # cancel because both columns are normalized.
    self_term = np.einsum("wt,wt->t", p_a, np.log(p_a))
    return self_term[:, None] - p_a.T @ np.log(p_b)


def _cross_jsd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # With m = (p + q) / 2 the "-x + y" terms of kl_div cancel between the two
    # halves, leaving JSD = 0.5 * sum xlogy(p, p/m) + 0.5 * sum xlogy(q, q/m).
    # The xlogy(x, x) parts depend on a single column each, so they are hoisted
    # out of the loop; only log(m) has to be recomputed per pair.
    self_a = xlogy(a, a).sum(axis=0)
    self_b = xlogy(b, b).sum(axis=0)
    dists = np.empty((a.shape[1], b.shape[1]), dtype=float)
    for col in range(b.shape[1]):
        other = b[:, [col]]
        mean = 0.5 * (a + other)
        # m is zero only where both columns are zero, and x * 0 == 0 there.
        log_mean = np.log(mean, where=mean > 0, out=np.zeros_like(mean))
        dists[:, col] = 0.5 * (self_a - (a * log_mean).sum(axis=0)) + 0.5 * (
            self_b[col] - (other * log_mean).sum(axis=0)
        )
    return dists


def _cross_bhat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # The scalar version clamps the *product* a * b, so every word where either
    # column is zero contributes sqrt(EPSILON) instead of zero.
    a_clean = np.where(np.isfinite(a), np.clip(a, 0.0, None), 0.0)
    b_clean = np.where(np.isfinite(b), np.clip(b, 0.0, None), 0.0)
    coefficient = np.sqrt(a_clean).T @ np.sqrt(b_clean)
    shared_support = (a_clean > 0).astype(float).T @ (b_clean > 0).astype(float)
    clamped = (a.shape[0] - shared_support) * np.sqrt(EPSILON)
    return -np.log(coefficient + clamped)


def _top_words_mask(values: np.ndarray, top_words: int) -> np.ndarray:
    """Boolean T x W matrix marking each column's ``top_words`` highest entries.

    ``argsort`` (rather than the faster ``argpartition``) is used so that ties are
    broken exactly as in :func:`_dist_jac`.
    """
    words_num, topics_num = values.shape
    count = min(top_words, words_num)
    top = np.argsort(values, axis=0)[-count:]
    mask = np.zeros((topics_num, words_num), dtype=bool)
    mask[np.repeat(np.arange(topics_num), count), top.T.ravel()] = True
    return mask


def _cross_jac(a: np.ndarray, b: np.ndarray, top_words: int = 100) -> np.ndarray:
    _validate_top_words(top_words)
    mask_a = _top_words_mask(a, top_words)
    mask_b = _top_words_mask(b, top_words)
    intersection = mask_a.astype(np.int32) @ mask_b.astype(np.int32).T
    union = mask_a.sum(axis=1)[:, None] + mask_b.sum(axis=1)[None, :] - intersection
    return 1 - intersection / union


def _cross_dists(
    a: np.ndarray, b: np.ndarray, method: str = "sklb", **kwargs
) -> np.ndarray:
    """Distances between every column of ``a`` and every column of ``b``.

    Vectorized counterpart of the scalar ``_dist_*`` functions, which remain the
    reference implementation. Returns an array of shape
    ``(a.shape[1], b.shape[1])`` where entry ``[i, j]`` is the distance from
    ``a[:, i]`` to ``b[:, j]``.
    """
    if method not in DIST_FUNCS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from {sorted(DIST_FUNCS)}"
        )

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if method == "jac":
        return _cross_jac(a, b, **kwargs)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"unexpected keyword arguments for method {method!r}: {unexpected}"
        )

    if method == "klb":
        return _cross_klb(a, b)
    if method in ("sklb", "jef"):
        # Jeffrey's divergence equals the symmetric KL divergence for
        # normalized distributions.
        return _cross_klb(a, b) + _cross_klb(b, a).T
    if method == "jsd":
        return _cross_jsd(a, b)
    if method == "hel":
        root_a = np.sqrt(_sanitize_columns(a))
        root_b = np.sqrt(_sanitize_columns(b))
        return distance.cdist(root_a.T, root_b.T, "euclidean") / np.sqrt(2)
    if method == "bhat":
        return _cross_bhat(a, b)
    if method == "tv":
        return distance.cdist(a.T, b.T, "cityblock") / 2
    raise AssertionError(f"validated distance method {method!r} was not handled")


def get_topics_dist(
    phi: Union[np.ndarray, DataFrame], method: str = "sklb", **kwargs
) -> np.ndarray:
    """Finding closest topics in models.

    Parameters
    ----------
    phi : Union[ndarray, DataFrame]
        Words vs topics matrix (W x T).
    method : str = "sklb"
        Comparison method. Possible variants:
        1) "klb" - Kullback-Leibler divergence.
        2) "sklb" - Symmetric Kullback-Leibler divergence.
        3) "jsd" - Jensen-Shannon divergence.
        4) "jef" - Jeffrey's divergence.
        5) "hel" - Hellinger distance.
        6) "bhat" - Bhattacharyya distance.
        7) "tv" — Total variation distance.
        8) "jac" - Jaccard index.
    **kwargs : dict
        Keyword arguments passed to distance function.

    Returns
    -------
    numpy.ndarray
        Topics distances matrix.
    """
    phi_copy = np.array(phi)

    if phi_copy.ndim != 2:
        raise ValueError("phi must be a 2D array (words x topics)")
    if np.any(phi_copy < 0):
        raise ValueError("phi must contain non-negative probability values")
    if not np.allclose(phi_copy.sum(axis=0), 1.0, atol=1e-6):
        raise ValueError("phi columns must sum to 1 (probability distributions)")

    topics_dists = _cross_dists(phi_copy, phi_copy, method, **kwargs)

    # Asymmetric divergences (e.g. "klb") are mirrored across the diagonal: the
    # value computed for the pair (i, j) with i < j is stored in both [i, j] and
    # [j, i]. Downstream consumers such as get_topics_scatter require a
    # symmetric matrix.
    upper = np.triu(topics_dists, 1)
    return upper + upper.T


def _classical_mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    count = distances.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    positive = eigenvalues > np.finfo(float).eps
    coords = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])

    # LocallyLinearEmbedding rejects an input with fewer dimensions than it is
    # asked to produce. A degenerate distance matrix - identical topics, or a
    # model that never separated - leaves fewer positive eigenvalues than that,
    # so pad with zero columns instead of handing over a narrower matrix.
    if coords.shape[1] < n_components:
        padding = np.zeros((count, n_components - coords.shape[1]))
        coords = np.hstack([coords, padding])
    return coords


def get_topics_scatter(
    topic_dists: np.ndarray,
    theta: np.ndarray,
    method: str = "tsne",
    method_kws: Optional[dict] = None,
) -> DataFrame:
    """Calculate topics coordinates for a scatter plot.

    Parameters
    ----------
    topic_dists : numpy.ndarray
        Topics distance matrix.
    theta : numpy.ndarray
        Topics vs documents probability matrix.
    method : str = 'tsne'
        Method to calculate topics scatter coordinates (X and Y).
        Possible values:
        1) 'tsne' - t-distributed Stochastic Neighbor Embedding.
        2) 'sem' - SpectralEmbedding.
        3) 'mds' - MDS.
        4) 'lle' - LocallyLinearEmbedding.
        5) 'ltsa' - LocallyLinearEmbedding with LTDA method.
        6) 'isomap' - Isomap.
    method_kws : dict = None
        Keyword arguments passed to method function.

    Returns
    -------
    DataFrame
        Topics scatter coordinates.
    """
    topic_dists = np.asarray(topic_dists, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if topic_dists.ndim != 2 or topic_dists.shape[0] != topic_dists.shape[1]:
        raise ValueError("topic_dists must be a square 2D matrix")
    if not np.isfinite(topic_dists).all():
        raise ValueError("topic_dists must contain only finite values")
    if not np.allclose(topic_dists, topic_dists.T):
        raise ValueError("topic_dists must be symmetric")
    if theta.ndim != 2 or theta.shape[0] != topic_dists.shape[0]:
        raise ValueError("theta topics dimension must match topic_dists")
    if topic_dists.shape[0] < 2:
        raise ValueError("at least two topics are required for scatter coordinates")

    if method not in SCATTER_METHODS:
        raise ValueError(
            f"Unknown scatter method {method!r}; choose from {SCATTER_METHODS}"
        )

    if topic_dists.shape[0] == 2:
        half_distance = topic_dists[0, 1] / 2
        coords = np.array([[-half_distance, 0.0], [half_distance, 0.0]])
        topics_xy = DataFrame(coords, columns=Index(["x", "y"]))
        topics_xy["topic"] = topics_xy.index.astype(int)
        topics_xy["size"] = calc_topics_marg_probs(theta) * 100
        return topics_xy

    method_kws = dict(method_kws or {})
    method_kws.setdefault("n_components", 2)

    # Most methods consume the distance matrix directly; the branches below
    # override this when a method needs a different representation.
    transform_input = topic_dists

    if method == "tsne":
        method_kws.setdefault("metric", "precomputed")
        method_kws.setdefault("init", "random")
        method_kws.setdefault("learning_rate", "auto")
        method_kws.setdefault("perplexity", min(50, max(topic_dists.shape[0] // 2, 1)))
        transformer = TSNE(**method_kws)

    elif method == "sem":
        method_kws.setdefault("affinity", "precomputed")
        transformer = SpectralEmbedding(**method_kws)
        nonzero = topic_dists[topic_dists > 0]
        scale = np.median(nonzero) if nonzero.size else 1.0
        transform_input = np.exp(-((topic_dists / scale) ** 2))
        np.fill_diagonal(transform_input, 1.0)

    elif method == "mds":
        mds_params = signature(MDS.__init__).parameters
        if "metric_mds" in mds_params:
            # scikit-learn >= 1.9 deprecated `dissimilarity` in favor of `metric`
            method_kws.setdefault("metric", "precomputed")
        else:
            method_kws.setdefault("dissimilarity", "precomputed")
        method_kws.setdefault("normalized_stress", "auto")
        method_kws.setdefault("n_init", 1)
        if "init" in mds_params:
            # `init` is a constructor argument only since scikit-learn 1.9
            method_kws.setdefault("init", "random")
        transformer = MDS(**method_kws)

    elif method == "lle":
        method_kws["method"] = "standard"
        method_kws.setdefault("n_neighbors", min(5, topic_dists.shape[0] - 1))
        transformer = LocallyLinearEmbedding(**method_kws)
        transform_input = _classical_mds(topic_dists, method_kws["n_components"])

    elif method == "ltsa":
        method_kws["method"] = "ltsa"
        method_kws.setdefault("n_neighbors", min(5, topic_dists.shape[0] - 1))
        transformer = LocallyLinearEmbedding(**method_kws)
        transform_input = _classical_mds(topic_dists, method_kws["n_components"])

    elif method == "isomap":
        method_kws.setdefault("metric", "precomputed")
        method_kws.setdefault("n_neighbors", min(5, topic_dists.shape[0] - 1))
        transformer = Isomap(**method_kws)

    else:
        raise AssertionError("validated scatter method was not handled")

    coords = transformer.fit_transform(transform_input)

    topics_xy = DataFrame(coords, columns=Index(["x", "y"]))
    topics_xy["topic"] = topics_xy.index.astype(int)
    # calc_topics_marg_probs already rejects an all-zero theta and returns
    # probabilities summing to 1, so scaling to percentages is unconditional.
    topics_xy["size"] = calc_topics_marg_probs(theta) * 100
    return topics_xy


def get_top_topic_words(
    phi: DataFrame,
    words_num: int = 20,
    topics_idx: Optional[Union[List[int], np.ndarray]] = None,
) -> DataFrame:
    """Select top topic words from a fitted model.

    Parameters
    ----------
    phi : pandas.DataFrame
        Words vs topics matrix (phi) with words as
        indices and topics as columns.
    words_num : int = 20
        The number of words to select.
    topics_idx : Union[List, numpy.ndarray] = None
        Topics indices.

    Returns
    -------
    DataFrame
        Words with highest probabilities in all (or selected) topics.
    """
    selected_topics = phi.columns if topics_idx is None else topics_idx
    return phi.loc[:, selected_topics].apply(
        lambda x: x.sort_values(ascending=False).head(words_num).index, axis=0
    )
