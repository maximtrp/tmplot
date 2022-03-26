"""Metrics module"""
from math import log
import numpy as np


def entropy(phi: np.ndarray):
    """Renyi entropy calculation routine [1]_.

    Renyi entropy can be used to estimate the optimal number of topics: fit
    several models varying the number of topics and choose the model for
    which Renyi entropy is minimal.

    Parameters
    ----------
    phi : np.ndarray
        Topics vs words probabilities matrix (T x W).

    Returns
    -------
    renyi : double
        Renyi entropy value.

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
    >>> # Entropy calculation
    >>> entropy = tmp.entropy(phi)
    """
    # Terms number
    words_num = phi.shape[1]
    # Topics number
    topics_num = phi.shape[0]

    # Setting threshold
    thresh = 1 / words_num

    # Maximum probability of each word
    p_max = np.zeros(words_num)

    # Obtaining maximum p value over all topics for each word
    p_max = np.max(phi, axis=0)

    # Select the probabilities larger than thresh
    p_max_mask = p_max > thresh
    word_ratio = p_max_mask.sum()
    sum_prob = p_max[p_max_mask].sum()

    # Shannon entropy
    shannon = log(word_ratio / (words_num * topics_num))

    # Internal energy
    int_energy = -log(sum_prob / topics_num)

    # Free energy
    free_energy = int_energy - shannon * topics_num

    # Renyi entropy
    if topics_num == 1:
        renyi = free_energy / topics_num
    else:
        renyi = free_energy / (topics_num-1)

    return renyi
