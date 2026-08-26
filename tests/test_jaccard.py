from unittest.mock import patch

import numpy as np
from pandas import DataFrame

from src import tmplot as tm


def test_jaccard_distance():
    first = np.array([0.7, 0.2, 0.1])
    same_ranking = np.array([0.6, 0.3, 0.1])
    different_top_word = np.array([0.2, 0.7, 0.1])

    assert tm._distance._dist_jac(first, same_ranking, top_words=1) == 0
    assert tm._distance._dist_jac(first, different_top_word, top_words=1) == 1


def test_jaccard_closest_topics_selects_smallest_distance():
    reference_phi = DataFrame(
        {0: [0.8, 0.1, 0.1], 1: [0.1, 0.8, 0.1]},
        index=["alpha", "beta", "gamma"],
    )
    current_phi = DataFrame(
        {0: [0.1, 0.8, 0.1], 1: [0.8, 0.1, 0.1]},
        index=["alpha", "beta", "gamma"],
    )

    with patch(
        "src.tmplot._stability.get_phi",
        side_effect=[reference_phi, current_phi],
    ):
        closest, distances = tm.get_closest_topics(
            [object(), object()], method="jac", top_words=1, verbose=False
        )

    assert closest[:, 1].tolist() == [1, 0]
    assert distances[:, 1].tolist() == [0.0, 0.0]


def test_jaccard_closest_topics_honors_top_words():
    reference_phi = DataFrame({0: [0.6, 0.3, 0.1]}, index=["a", "b", "c"])
    current_phi = DataFrame(
        {0: [0.1, 0.6, 0.3], 1: [0.5, 0.1, 0.4]},
        index=["a", "b", "c"],
    )

    with patch(
        "src.tmplot._stability.get_phi",
        side_effect=[reference_phi, current_phi],
    ):
        closest, _ = tm.get_closest_topics(
            [object(), object()], method="jac", top_words=1, verbose=False
        )

    assert closest[0, 1] == 1
