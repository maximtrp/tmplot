__all__ = ["get_topics_dist", "get_topics_scatter", "get_top_topic_words"]
from typing import Optional, Union, List
from itertools import combinations
from pandas import DataFrame, Index
import numpy as np
from scipy.special import kl_div
from scipy.spatial import distance
from sklearn.manifold import (
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
)
from ._helpers import calc_topics_marg_probs


def _dist_klb(a1: np.ndarray, a2: np.ndarray):
    dist = kl_div(a1, a2)
    return dist[np.isfinite(dist)].sum()


def _dist_sklb(a1: np.ndarray, a2: np.ndarray):
    dist = kl_div(a1, a2) + kl_div(a2, a1)
    return dist[np.isfinite(dist)].sum()


def _dist_jsd(a1: np.ndarray, a2: np.ndarray):
    m = 0.5 * (a1 + a2)
    dist = 0.5 * kl_div(a1, m) + 0.5 * kl_div(a2, m)
    return dist[np.isfinite(dist)].sum()


def _dist_jef(a1: np.ndarray, a2: np.ndarray):
    vals = (a1 - a2) * (np.log(a1) - np.log(a2))
    vals[(vals <= 0) | ~np.isfinite(vals)] = 0.0
    return vals.sum()


# Small value for numerical stability
EPSILON = 1e-64


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
    a = np.argsort(a1)[: -top_words - 1 : -1]
    b = np.argsort(a2)[: -top_words - 1 : -1]
    j_num = np.intersect1d(a, b, assume_unique=False).size
    j_den = np.union1d(a, b).size
    jac_val = 1 - j_num / j_den
    return jac_val


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

    topics_num = phi_copy.shape[1]
    topics_pairs = combinations(range(topics_num), 2)

    # Topics distances matrix
    topics_dists = np.zeros(shape=(topics_num, topics_num), dtype=float)

    dist_funcs = {
        "klb": _dist_klb,
        "sklb": _dist_sklb,
        "jsd": _dist_jsd,
        "jef": _dist_jef,
        "hel": _dist_hel,
        "bhat": _dist_bhat,
        "tv": _dist_tv,
        "jac": _dist_jac,
    }

    for i, j in topics_pairs:
        _dist_func = dist_funcs.get(method, "sklb")
        topics_dists[((i, j), (j, i))] = _dist_func(
            phi_copy[:, i], phi_copy[:, j], **kwargs
        )

    return topics_dists


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
    if not method_kws:
        method_kws = {"n_components": 2}

    if method == "tsne":
        method_kws.setdefault("init", "pca")
        method_kws.setdefault("learning_rate", "auto")
        method_kws.setdefault("perplexity", min(50, max(topic_dists.shape[0] // 2, 1)))
        transformer = TSNE(**method_kws)

    elif method == "sem":
        method_kws.setdefault("affinity", "precomputed")
        transformer = SpectralEmbedding(**method_kws)

    elif method == "mds":
        method_kws.setdefault("dissimilarity", "precomputed")
        method_kws.setdefault("normalized_stress", "auto")
        method_kws.setdefault("n_init", 1)
        transformer = MDS(**method_kws)

    elif method == "lle":
        method_kws["method"] = "standard"
        transformer = LocallyLinearEmbedding(**method_kws)

    elif method == "ltsa":
        method_kws["method"] = "ltsa"
        transformer = LocallyLinearEmbedding(**method_kws)

    elif method == "isomap":
        transformer = Isomap(**method_kws)

    coords = transformer.fit_transform(topic_dists)

    topics_xy = DataFrame(coords, columns=Index(["x", "y"]))
    topics_xy["topic"] = topics_xy.index.astype(int)
    topics_xy["size"] = calc_topics_marg_probs(theta)
    size_sum = topics_xy["size"].sum()
    if size_sum > 0:
        topics_xy["size"] *= 100 / topics_xy["size"].sum()
    else:
        topics_xy["size"] = np.nan
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
    return phi.loc[:, topics_idx or phi.columns].apply(
        lambda x: x.sort_values(ascending=False).head(words_num).index, axis=0
    )
