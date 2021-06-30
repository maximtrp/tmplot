# TODO: top docs in topic
# TODO: stable topics
__all__ = [
    'get_topics_dist', 'get_topics_scatter', 'get_top_topic_words']
from typing import Union, List
from itertools import combinations
from pandas import DataFrame
import numpy as np
from scipy.special import kl_div
from scipy.spatial import distance
from sklearn.manifold import (
    TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding)
from ._helpers import calc_topics_marg_probs


def _dist_klb(a1: np.ndarray, a2: np.ndarray):
    dist = kl_div(a1, a2)
    return dist[np.isfinite(dist)].sum()


def _dist_sklb(a1: np.ndarray, a2: np.ndarray):
    dist = kl_div(a1, a2) + kl_div(a1, a2)
    return dist[np.isfinite(dist)].sum()


def _dist_jsd(a1: np.ndarray, a2: np.ndarray):
    dist = 0.5 * kl_div(a1, a2) + 0.5 * kl_div(a1, a2)
    return dist[np.isfinite(dist)].sum()


def _dist_jef(a1: np.ndarray, a2: np.ndarray):
    vals = (a1 - a2) * (np.log(a1) - np.log(a2))
    vals[(vals <= 0) | ~np.isfinite(vals)] = 0.
    return vals.sum()


def _dist_hel(a1: np.ndarray, a2: np.ndarray):
    a1[(a1 <= 0) | ~np.isfinite(a1)] = 1e-64
    a2[(a2 <= 0) | ~np.isfinite(a2)] = 1e-64
    hel_val = distance.euclidean(
        np.sqrt(a1), np.sqrt(a2)) / np.sqrt(2)
    return hel_val


def _dist_bhat(a1: np.ndarray, a2: np.ndarray):
    pq = a1 * a2
    pq[(pq <= 0) | ~np.isfinite(pq)] = 1e-64
    dist = -np.log(np.sum(np.sqrt(pq)))
    return dist


def _dist_jac(a1: np.ndarray, a2: np.ndarray,  top_words=100):
    a = np.argsort(a1)[:-top_words-1:-1]
    b = np.argsort(a2)[:-top_words-1:-1]
    j_num = np.intersect1d(a, b, assume_unique=False).size
    j_den = np.union1d(a, b).size
    jac_val = 1 - j_num / j_den
    return jac_val


def get_topics_dist(
        phi: Union[np.ndarray, DataFrame],
        method: str = "sklb",
        **kwargs) -> np.ndarray:
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
        7) "jac" - Jaccard index.
    **kwargs : dict
        Keyword arguments passed to distance function.

    Returns
    -------
    numpy.ndarray
        Topics distances matrix.
    """
    phi_copy = np.array(phi)
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
        "jac": _dist_jac,
    }

    for i, j in topics_pairs:
        _dist_func = dist_funcs.get(method, "sklb")
        topics_dists[((i, j), (j, i))] = _dist_func(
            phi_copy[:, i], phi_copy[:, j], **kwargs)

    return topics_dists


def get_topics_scatter(
        topic_dists: np.ndarray,
        theta: np.ndarray,
        method: str = 'tsne',
        method_kws: dict = None) -> DataFrame:
    """Calculate topics coordinates for a scatter plot.

    Parameters
    ----------
    topic_dists : numpy.ndarray
        Topics distance matrix.
    theta : numpy.ndarray
        Topics vs documents probability matrix.
    method : str = 'graph'
        Method to calculate topics scatter coordinates (X and Y).
        Possible values:
        1) 'tsne' - t-distributed Stochastic Neighbor Embedding.
        2) 'sem' - SpectralEmbedding.
        3) 'mds' - MDS.
        4) 'lle' - LocallyLinearEmbedding.
        5) 'isomap' - Isomap.
    method_kws : dict = None
        Keyword arguments passed to method function.

    Returns
    -------
    DataFrame
        Topics scatter coordinates.
    """
    if not method_kws:
        method_kws = {'n_components': 2}

    if method == 'tsne':
        method_kws.setdefault('metric',  'precomputed')
        transformer = TSNE(**method_kws)

    elif method == 'sem':
        method_kws.setdefault('affinity', 'precomputed')
        transformer = SpectralEmbedding(**method_kws)

    elif method == 'mds':
        method_kws.setdefault('dissimilarity', 'precomputed')
        transformer = MDS(**method_kws)

    elif method == 'lle':
        transformer = LocallyLinearEmbedding(**method_kws)

    elif method == 'isomap':
        transformer = Isomap(**method_kws)

    coords = transformer.fit_transform(topic_dists)

    topics_xy = DataFrame(coords, columns=['x', 'y'])
    topics_xy['topic'] = topics_xy.index.astype(int)
    topics_xy['size'] = calc_topics_marg_probs(theta)
    topics_xy['size'] *= (100 / topics_xy['size'].sum())
    return topics_xy


def get_top_topic_words(
        phi: DataFrame,
        words_num: int = 20,
        topics_idx: Union[List[int], np.ndarray] = None) -> DataFrame:
    """Select top topic words from a fitted model.

    Parameters
    ----------
    phi : DataFrame
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
    return phi.loc[:, topics_idx or phi.columns]\
        .apply(
            lambda x: x
            .sort_values(ascending=False)
            .head(words_num).index, axis=0
        )
