from __future__ import annotations

__all__ = ["get_closest_topics", "get_stable_topics"]
from typing import Any

import numpy as np
import tqdm

from ._distance import DIST_FUNCS, _cross_dists
from ._helpers import get_phi


def get_closest_topics(
    models: list[Any],
    ref: int = 0,
    method: str = "sklb",
    top_words: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
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

    if models_num == 0:
        raise ValueError("at least one model is required")
    if not 0 <= ref < models_num:
        raise ValueError(f"ref must be in [0, {models_num - 1}], got {ref}")
    if method not in DIST_FUNCS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from {sorted(DIST_FUNCS)}"
        )

    # Reference model
    model_ref = models[ref]

    # Words vs topics matrix (phi)
    model_ref_phi = get_phi(model_ref)

    # Number of topics
    topics_num = model_ref_phi.shape[1]

    # Array with the closest topics ids
    closest_topics = np.zeros(shape=(topics_num, models_num), dtype=int)
    closest_topics[:, ref] = np.arange(topics_num)

    dist_kwargs = {"top_words": top_words} if method == "jac" else {}

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

        # Distance matrix for all topic pairs
        all_vs_all_dists = _cross_dists(
            ref_phi.to_numpy(), current_phi.to_numpy(), method, **dist_kwargs
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
) -> tuple[np.ndarray, np.ndarray]:
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
    closest_topics = np.asarray(closest_topics)
    dist_arr = np.asarray(dist, dtype=float)

    if closest_topics.ndim != 2 or dist_arr.ndim != 2:
        raise ValueError("closest_topics and dist must both be 2D arrays")
    if closest_topics.shape != dist_arr.shape:
        raise ValueError(
            "closest_topics and dist must have the same shape, got "
            f"{closest_topics.shape} and {dist_arr.shape}"
        )
    if not 0 <= ref < dist_arr.shape[1]:
        raise ValueError(f"ref must be in [0, {dist_arr.shape[1] - 1}], got {ref}")

    max_dist = dist_arr.max()
    dist_ready = dist_arr / max_dist if norm and max_dist > 0 else dist_arr.copy()
    dist_ready = inverse_factor - dist_ready if inverse else dist_ready
    mask = np.sum(np.delete(dist_ready, ref, axis=1) >= thres, axis=1) >= thres_models
    return closest_topics[mask], dist_ready[mask]
