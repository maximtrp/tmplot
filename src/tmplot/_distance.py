# TODO: calculate distances between topics using a number of distance metrics
# TODO: top words in topic
# TODO: top docs in topic
# TODO: word salience?
# TODO: stable topics
import numpy as np
import tqdm
from pandas import DataFrame
from typing import Union, Tuple, List
from scipy.special import kl_div
from scipy.spatial import distance
from itertools import combinations
from networkx import from_numpy_matrix, spring_layout


def get_topics_dist(
        phi: Union[np.ndarray, DataFrame],
        ref: int = 0,
        method: str = "sklb",
        thres: float = 0.9,
        top_words: int = 100,
        verbose: bool = True) -> np.ndarray:
    """Finding closest topics in models.

    Parameters
    ----------
    phi : Union[ndarray, DataFrame]
        Sequence of words vs topics matrices (W x T).
    ref : int = 0
        Index of reference matrix (zero-based indexing).
    method : str = "sklb"
        Comparison method. Possible variants:
        1) "klb" - Kullback-Leibler divergence.
        2) "sklb" - Symmetric Kullback-Leibler divergence.
        3) "jsd" - Jensen-Shannon divergence.
        4) "jef" - Jeffrey's divergence.
        5) "hel" - Hellinger distance.
        6) "bhat" - Bhattacharyya distance.
        6) "jac" - Jaccard index.
    thres : float = 0.9
        Threshold for topic filtering.
    top_words : int = 100
        Number of top words in each topic to use in Jaccard index calculation.
    verbose : bool = True
        Verbose output (progress bar).

    Returns
    -------
    dist : numpy.ndarray
        Topics distances matrix.

    Example
    -------
    """
    topics_num = phi.shape[1]
    topics_pairs = combinations(range(topics_num), 2)

    # Topics distances matrix
    topics_dist = np.zeros(shape=(topics_num, topics_num), dtype=float)

    def enum_func(x):
        return tqdm.tqdm(x) if verbose else x

    for i, j in enum_func(topics_pairs):

        if method == "klb":
            val_raw = kl_div(phi[:, i], phi[:, j])
            topics_dist[i, j] = val_raw[np.isfinite(val_raw)].sum()

        elif method == "sklb":
            val_raw = kl_div(phi[:, i], phi[:, j])\
                + kl_div(phi[:, j], phi[:, i])
            topics_dist[i, j] = val_raw[np.isfinite(val_raw)].sum()

        elif method == "jsd":
            val_raw = 0.5 * kl_div(phi[:, i], phi[:, j])\
                + 0.5 * kl_div(phi[:, j], phi[:, i])
            topics_dist[i, j] = val_raw[np.isfinite(val_raw)].sum()

        elif method == "jef":
            p = phi[:, i]
            q = phi[:, j]
            vals = (p - q) * (np.log(p) - np.log(q))
            vals[(vals <= 0) | ~np.isfinite(vals)] = 0.
            topics_dist[i, j] = vals.sum()

        elif method == "hel":
            p = phi[:, i]
            q = phi[:, j]
            p[(p <= 0) | ~np.isfinite(p)] = 1e-64
            q[(q <= 0) | ~np.isfinite(q)] = 1e-64
            hel_val = distance.euclidean(
                np.sqrt(p), np.sqrt(q)) / np.sqrt(2)
            topics_dist[i, j] = hel_val

        elif method == "bhat":
            p = phi[:, i]
            q = phi[:, j]
            pq = p * q
            pq[(pq <= 0) | ~np.isfinite(pq)] = 1e-64
            dist = -np.log(np.sum(np.sqrt(pq)))
            topics_dist[i, j] = dist

        elif method == "jac":
            a = np.argsort(phi[:, i])[:-top_words-1:-1]
            b = np.argsort(phi[:, j])[:-top_words-1:-1]
            j_num = np.intersect1d(a, b, assume_unique=False).size
            j_den = np.union1d(a, b).size
            jac_val = j_num / j_den
            topics_dist[i, j] = jac_val

    return topics_dist


def get_topics_scatter(
    tdm: np.ndarray,
    method: str = 'graph',
    method_kws: dict = {}) -> DataFrame:
    """Calculating topics coordinates for a scatterplot.

    Parameters
    ----------
    tdm : numpy.ndarray
        Topics distance matrix.
    """
    if method == 'graph':
        g = from_numpy_matrix(tdm)
        layout = spring_layout(g, **method_kws)
        coords = np.array(list(layout.values()))
        coords = np.hstack((coords, np.arange(1, coords.shape[0] + 1)[:, None]))
    elif method == 'pca':
        pass

    topics_xy = DataFrame(coords, columns=['x', 'y', 'topic'])
    topics_xy['topic'] = topics_xy['topic'].astype(int)
    return topics_xy

# def get_stable_topics(
#         closest_topics: np.ndarray,
#         dist: np.ndarray,
#         norm: bool = True,
#         inverse: bool = True,
#         inverse_factor: float = 1.0,
#         ref: int = 0,
#         thres: float = 0.9,
#         thres_models: int = 2) -> Tuple[np.ndarray, np.ndarray]:
#     """Finding stable topics in models.

#     Parameters
#     ----------
#     closest_topics : np.ndarray
#         Closest topics indices in a two-dimensional array.
#         Columns correspond to the compared matrices (their indices),
#         rows are the closest topics pairs. Typically, this should be
#         the first value returned by :meth:`bitermplus.get_closest_topics`
#         function.
#     dist : np.ndarray
#         Distance values: Kullback-Leibler divergence or Jaccard index values
#         corresponding to the matrix of the closest topics.
#         Typically, this should be the second value returned by
#         :meth:`bitermplus.get_closest_topics` function.
#     norm : bool = True
#         Normalize distance values (passed as ``dist`` argument).
#     inverse : bool = True
#         Inverse distance values by subtracting them from ``inverse_factor``.
#     inverse_factor : float = 1.0
#         Subtract distance values from this factor to inverse.
#     ref : int = 0
#         Index of reference matrix (i.e. reference column index,
#         zero-based indexing).
#     thres : float = 0.9
#         Threshold for distance values filtering.
#     thres_models : int = 2
#         Minimum topic recurrence frequency across all models.

#     Returns
#     -------
#     stable_topics : np.ndarray
#         Filtered matrix of the closest topics indices (i.e. stable topics).
#     dist : np.ndarray
#         Filtered distance values corresponding to the matrix of
#         the closest topics.

#     See Also
#     --------
#     bitermplus.get_closest_topics

#     Example
#     -------
#     >>> closest_topics, kldiv = btm.get_closest_topics(
#     ...     *list(map(lambda x: x.matrix_topics_words_, models)))
#     >>> stable_topics, stable_kldiv = btm.get_stable_topics(
#     ...     closest_topics, kldiv)
#     """
#     dist_arr = np.asarray(dist)
#     dist_ready = dist_arr / dist_arr.max() if norm else dist_arr.copy()
#     dist_ready = inverse_factor - dist_ready if inverse else dist_ready
#     mask = (
#         np.sum(np.delete(dist_ready, ref, axis=1) >= thres, axis=1)
#         >= thres_models)
#     return closest_topics[mask], dist_ready[mask]


def get_top_topic_words(
        phi: DataFrame,
        words_num: int = 20,
        topics_loc: Union[List[int], np.ndarray] = None) -> DataFrame:
    """Select top topic words from a fitted model.

    Parameters
    ----------
    phi : DataFrame
        Words vs topics matrix with words as indices and topics as columns.
    words_num : int = 20
        The number of words to select.
    topics_loc : Union[List, numpy.ndarray] = None
        Topics indices. Meant to be used to select only stable
        topics.

    Returns
    -------
    DataFrame
        Words with highest probabilities in all selected topics.

    Example
    -------
    """
    return phi.loc[:, topics_loc or phi.columns]\
        .apply(
            lambda x: x\
                .sort_values(ascending=False)\
                .head(words_num).index, axis=0
        )


# def get_top_topic_docs(
#         docs: Union[List[str], np.ndarray],
#         p_zd: np.ndarray,
#         docs_num: int = 20,
#         topics_idx: Union[List[int], np.ndarray] = None) -> DataFrame:
#     """Select top topic docs from a fitted model.

#     Parameters
#     ----------
#     docs : Union[List[str], np.ndarray]
#         Iterable of documents (e.g. list of strings).
#     p_zd : np.ndarray
#         Documents vs topics probabilities matrix.
#     docs_num : int = 20
#         The number of documents to select.
#     topics_idx : Union[List, numpy.ndarray] = None
#         Topics indices. Meant to be used to select only stable
#         topics.

#     Returns
#     -------
#     DataFrame
#         Documents with highest probabilities in all selected topics.

#     Example
#     -------
#     >>> top_docs = btm.get_top_topic_docs(
#     ...     texts,
#     ...     p_zd,
#     ...     docs_num=100,
#     ...     topics_idx=[1,2,3,4])
#     """
#     def _select_docs(docs, p_zd, topic_id: int):
#         ps = p_zd[:, topic_id]
#         idx = np.argsort(ps)[:-docs_num-1:-1]
#         result = Series(np.asarray(docs)[idx])
#         result.name = 'topic{}'.format(topic_id)
#         return result

#     topics_num = p_zd.shape[1]
#     topics_idx = np.arange(topics_num) if topics_idx is None else topics_idx
#     return concat(
#         map(lambda x: _select_docs(docs, p_zd, x), topics_idx), axis=1)
