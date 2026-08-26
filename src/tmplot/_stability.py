__all__ = ["get_closest_topics", "get_stable_topics"]
from typing import List, Tuple, Any
import numpy as np
import tqdm
from ._distance import (
    _dist_klb,
    _dist_sklb,
    _dist_jsd,
    _dist_jef,
    _dist_hel,
    _dist_bhat,
    _dist_jac,
    _dist_tv,
)
from ._helpers import get_phi

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


def get_closest_topics(
    models: List[Any],
    ref: int = 0,
    method: str = "sklb",
    top_words: int = 100,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finding closest topics in models.

    Parameters
    ----------
    models : List[Any]
        List of supported and fitted topic models.
    ref : int = 0
        Index of reference matrix (zero-based indexing).
    method : str = "sklb"
        Distance calculation method. Possible variants:
        1) "klb" - Kullback-Leibler divergence.
        2) "sklb" - Symmetric Kullback-Leibler divergence.
        3) "jsd" - Jensen-Shannon divergence.
        4) "jef" - Jeffrey's divergence.
        5) "hel" - Hellinger distance.
        6) "bhat" - Bhattacharyya distance.
        7) "tv" - Total variation distance.
        8) "jac" - Jaccard index.
    top_words : int = 100
        Number of top words in each topic to use in Jaccard index calculation.
    verbose : bool = True
        Verbose output (progress bar).

    Returns
    -------
    closest_topics : np.ndarray
        Closest topics indices in one two-dimensional array (topics ✕ models).
        Columns correspond to the compared models (their indices),
        rows are the closest topics pairs.
    dist : np.ndarray
        Closest topics distances (e.g., Kullback-Leibler or Jaccard index
        values). Shape of this array corresponds to the shape of the first
        returned argument.

    Example
    -------
    >>> # `models` must be an iterable of fitted models
    >>> closest_topics, kldiv = tmplot.get_closest_topics(models)
    """
    # Number of models passed
    models_num = len(models)

    # Reference model id
    ref = models_num - 1 if ref >= models_num else ref

    # Reference model
    model_ref = models[ref]

    # Words vs topics matrix (phi)
    model_ref_phi = get_phi(model_ref)

    # Number of topics
    topics_num = model_ref_phi.shape[1]

    # Array with the closest topics ids
    closest_topics = np.zeros(shape=(topics_num, models_num), dtype=int)
    closest_topics[:, ref] = np.arange(topics_num)

    if method not in dist_funcs:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from {sorted(dist_funcs)}"
        )
    dist_func = dist_funcs[method]

    # Distance values
    dist_vals = np.zeros(shape=(topics_num, models_num), dtype=float)

    def enum_func(x):
        return enumerate(tqdm.tqdm(x)) if verbose else enumerate(x)

    # Iterating over all models
    for mid, model in enum_func(models):
        # Current model is equal to reference model, skipping
        if mid == ref:
            continue

        # Get phi matrix for current model
        current_phi = get_phi(model)

        shared_words = model_ref_phi.index.intersection(current_phi.index)
        if shared_words.empty:
            raise ValueError("models do not have any vocabulary terms in common")
        ref_phi = model_ref_phi.loc[shared_words]
        current_phi = current_phi.loc[shared_words]
        ref_sums = ref_phi.sum(axis=0)
        current_sums = current_phi.sum(axis=0)
        if (ref_sums <= 0).any() or (current_sums <= 0).any():
            raise ValueError("shared vocabulary has zero probability mass for a topic")
        ref_phi = ref_phi / ref_sums
        current_phi = current_phi / current_sums
        current_topics_num = current_phi.shape[1]

        # Distance matrix for all topic pairs
        all_vs_all_dists = np.zeros((topics_num, current_topics_num))

        # Iterating over all topic pairs
        for t_ref in range(topics_num):
            for t in range(current_topics_num):
                kwargs = {"top_words": top_words} if method == "jac" else {}
                all_vs_all_dists[t_ref, t] = dist_func(
                    ref_phi.iloc[:, t_ref], current_phi.iloc[:, t], **kwargs
                )

        # Creating two arrays for the closest topics ids and distance values
        closest_topics[:, mid] = np.argmin(all_vs_all_dists, axis=1)
        dist_vals[:, mid] = np.min(all_vs_all_dists, axis=1)

    return closest_topics, dist_vals


def get_stable_topics(
    closest_topics: np.ndarray,
    dist: np.ndarray,
    norm: bool = True,
    inverse: bool = True,
    inverse_factor: float = 1.0,
    ref: int = 0,
    thres: float = 0.9,
    thres_models: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finding stable topics in models.

    Parameters
    ----------
    closest_topics : np.ndarray
        Closest topics indices in a two-dimensional array.
        Columns correspond to the compared matrices (their indices),
        rows are the closest topics pairs. Typically, this should be
        the first value returned by :meth:`tmplot.get_closest_topics`
        function.
    dist : np.ndarray
        Distance values: Kullback-Leibler divergence or Jaccard index values
        corresponding to the matrix of the closest topics.
        Typically, this should be the second value returned by
        :meth:`tmplot.get_closest_topics` function.
    norm : bool = True
        Normalize distance values (passed as ``dist`` argument).
    inverse : bool = True
        Inverse distance values by subtracting them from ``inverse_factor``.
        Should be set to ``False`` if Jaccard index was used to calculate
        closest topics.
    inverse_factor : float = 1.0
        Subtract distance values from this factor to inverse.
    ref : int = 0
        Index of reference matrix (i.e. reference column index,
        zero-based indexing).
    thres : float = 0.9
        Threshold for distance values filtering.
    thres_models : int = 2
        Minimum topic recurrence frequency across all models.

    Returns
    -------
    stable_topics : np.ndarray
        Filtered matrix of the closest topics indices (i.e. stable topics).
    dist : np.ndarray
        Filtered distance values corresponding to the matrix of
        the closest topics.

    See Also
    --------
    tmplot.get_closest_topics

    Example
    -------
    >>> closest_topics, kldiv = tmplot.get_closest_topics(models)
    >>> stable_topics, stable_kldiv = tmplot.get_stable_topics(
    ...     closest_topics, kldiv)
    """
    dist_arr = np.asarray(dist)
    max_dist = dist_arr.max()
    dist_ready = dist_arr / max_dist if norm and max_dist > 0 else dist_arr.copy()
    dist_ready = inverse_factor - dist_ready if inverse else dist_ready
    mask = np.sum(np.delete(dist_ready, ref, axis=1) >= thres, axis=1) >= thres_models
    return closest_topics[mask], dist_ready[mask]
