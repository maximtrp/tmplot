"""Metrics module"""
from __future__ import annotations

from math import log
from typing import Optional
from warnings import warn

import numpy as np

__all__ = ["entropy"]


def _resolve_topics_axis(
    phi: np.ndarray, max_probs: bool, topics_axis: Optional[int]
) -> int:
    """Determine (or verify) which axis of ``phi`` indexes topics.

    Each topic is a probability distribution over words, so the axis that sums
    to 1 is the *word* axis and the other one indexes topics. That is a defining
    property of phi rather than a heuristic, which makes it safe to rely on both
    for inferring the orientation and for checking a caller-supplied one:
    getting the orientation wrong would otherwise return a plausible but
    meaningless number instead of failing.
    """
    # phi is T x W when its rows sum to 1, W x T when its columns do.
    rows_normalized = np.allclose(phi.sum(axis=1), 1.0, atol=1e-6)
    cols_normalized = np.allclose(phi.sum(axis=0), 1.0, atol=1e-6)
    unverified = (
        "so the orientation of phi could not be verified; pass topics_axis "
        "explicitly and make sure each topic's word distribution sums to 1."
    )

    if topics_axis is None:
        if rows_normalized and cols_normalized:
            # Both axes summing to 1 forces phi to be square. The Renyi
            # calculation depends only on the shape and on the set of entries
            # above the threshold, so it is orientation invariant here - except
            # under max_probs, which reduces along a specific axis.
            if max_probs and not np.allclose(phi, phi.T):
                raise ValueError(
                    "cannot infer the topics axis: both axes of phi are "
                    "normalized and max_probs reduces along one of them; pass "
                    "topics_axis explicitly (0 for T x W, 1 for W x T)"
                )
            return 0
        if rows_normalized:
            return 0
        if cols_normalized:
            return 1
        warn(
            f"Neither axis of phi sums to 1, {unverified} Assuming a "
            "topics x words matrix.",
            UserWarning,
            stacklevel=3,
        )
        return 0

    if topics_axis not in (0, 1):
        raise ValueError(f"topics_axis must be 0 or 1, got {topics_axis!r}")

    # An explicit topics_axis is still checked against the same invariant.
    declared, opposite = (
        (rows_normalized, cols_normalized)
        if topics_axis == 0
        else (cols_normalized, rows_normalized)
    )
    if not declared and opposite:
        expected, actual = (
            ("T x W", "W x T") if topics_axis == 0 else ("W x T", "T x W")
        )
        raise ValueError(
            f"topics_axis={topics_axis} declares a {expected} matrix, but phi is "
            f"normalized along the other axis and so looks like {actual}; "
            f"pass topics_axis={1 - topics_axis} or transpose phi"
        )
    if not declared:
        warn(
            f"Neither axis of phi sums to 1, {unverified}",
            UserWarning,
            stacklevel=3,
        )
    return topics_axis


def entropy(
    phi: np.ndarray,
    max_probs: bool = False,
    topics_axis: Optional[int] = None,
):
    """Renyi entropy calculation routine [1]_.

    Renyi entropy can be used to estimate the optimal number of topics: fit
    several models varying the number of topics and choose the model for
    which Renyi entropy is minimal.

    Parameters
    ----------
    phi : np.ndarray
        Topics vs words probabilities matrix. Either orientation is accepted:
        T x W (topics in rows) or the W x T matrix returned by
        :meth:`tmplot.get_phi`. See ``topics_axis``.
    max_probs : bool
        Use maximum probabilities of terms per topics instead of all probability
        values.

        .. note::
           The default, ``False``, thresholds the whole matrix at ``1/W`` as the
           paper prescribes, and matches :func:`bitermplus.entropy` with its own
           default ``max_probs=True`` exactly. The two packages name this flag in
           opposite ways. Setting ``max_probs=True`` here selects a further
           variant, reducing to each word's largest probability across topics
           before thresholding.
    topics_axis : Optional[int]
        Axis along which topics are indexed: 0 for a T x W matrix, 1 for a
        W x T one. When ``None`` (default) it is inferred from which axis holds
        normalized probability distributions, since each topic's word
        distribution must sum to 1.

    Returns
    -------
    renyi : double
        Renyi entropy value.

    Raises
    ------
    ValueError
        If the matrix holds a single topic. Renyi entropy is undefined there:
        the deformation parameter ``q = 1/T`` equals 1, so the ``F / (q - 1)``
        denominator is zero.

    References
    ----------
    .. [1] Koltcov, S. (2018). Application of Rényi and Tsallis entropies to
           topic modeling optimization. Physica A: Statistical Mechanics and its
           Applications, 512, 1192-1204.

    Example
    -------
    >>> import tmplot as tmp
    >>> # Preprocessing step
    >>> # ...
    >>> # Model fitting step
    >>> # model = ...
    >>> # phi = ...
    >>> # Entropy calculation (orientation is detected automatically)
    >>> entropy = tmp.entropy(phi)
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2 or 0 in phi.shape:
        raise ValueError("phi must be a non-empty 2D topics x words matrix")
    if not np.isfinite(phi).all() or np.any(phi < 0):
        raise ValueError("phi must contain finite non-negative probabilities")

    topics_axis = _resolve_topics_axis(phi, max_probs, topics_axis)

    # Work internally on the T x W orientation.
    if topics_axis == 1:
        phi = phi.T

    if phi.shape[0] == 1:
        raise ValueError(
            "Renyi entropy is undefined for a single topic: q = 1/T = 1 makes "
            "the F / (q - 1) denominator zero. Compare two or more topics."
        )

    # Terms number
    words_num = phi.shape[1]
    # Topics number
    topics_num = phi.shape[0]

    # Setting threshold
    thresh = 1 / words_num

    if max_probs:
        # Obtaining maximum p value over all topics for each word
        p_max = np.max(phi, axis=0)

        # Select the probabilities larger than thresh
        p_max_mask = p_max >= thresh
        word_ratio = p_max_mask.sum()
        sum_prob = p_max[p_max_mask].sum()

    else:
        # Select the probabilities larger than thresh
        mask = phi >= thresh
        sum_prob = phi[mask].sum()
        word_ratio = np.count_nonzero(mask)

    if word_ratio == 0 or sum_prob <= 0:
        raise ValueError("phi does not contain probabilities at or above the threshold")

    # Shannon entropy
    shannon = log(word_ratio / (words_num * topics_num))

    # Internal energy
    int_energy = -log(sum_prob / topics_num)

    # Note this quantity is -T * F, not the free energy F itself; the trailing
    # division by T - 1 cancels the extra factor. Named as in bitermplus 1.0.
    neg_scaled_free_energy = int_energy - shannon * topics_num

    # Renyi entropy
    return neg_scaled_free_energy / (topics_num - 1)
