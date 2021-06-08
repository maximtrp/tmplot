__all__ = ['get_phi', 'get_theta', 'calc_terms_relevance', 'calc_topics_probs']
from numpy import ndarray
from pandas import concat, Series, DataFrame
from tomotopy import LDAModel, LLDAModel
from functools import partial
from joblib import delayed, Parallel
from typing import Union


def get_phi(
        model: object) -> ndarray:
    """Returns topics (T) vs words (W) matrix of shape (T, W)."""

    tomotopy_models = [LDAModel, LLDAModel]

    if any(map(partial(isinstance, model), tomotopy_models)):

        # Topics vs words distributions
        twd = map(
            lambda x: Series(model.get_topic_word_dist(x)),
            range(model.k))

        # Concatenating into DataFrame
        phi = concat(twd, axis=1)

        # Settings terms from vocabulary as index
        phi.index = model.vocabs

    return phi


def get_theta(
        model: object) -> DataFrame:
    tomotopy_models = [LDAModel, LLDAModel]

    if any(map(partial(isinstance, model), tomotopy_models)):
        tdd = map(lambda x: Series(x.get_topic_dist()), model.docs)
        theta = concat(tdd, axis=1)

    return theta


def calc_topics_marg_probs(
        theta: DataFrame):
    """Calculate marginal topics probabilities"""
    return theta.sum(axis=1)


def calc_terms_marg_probs(
        phi: Union[ndarray, DataFrame]) -> Union[ndarray, Series]:
    """Calculate marginal terms probabilities.

    Parameters
    ----------
    phi : Union[ndarray, DataFrame]
        Words vs topics matrix (W x T).

    Returns
    -------
    Union[ndarray, Series]
        Marginal terms probabilities.
    """
    return phi.sum(axis=1)


def calc_terms_salience(
        terms_freqs: ndarray,
        phi: ndarray,
        ):
    # saliency(term w) = frequency(w) * [sum_t p(t | w) * log(p(t | w)/p(t))] for topics t
    # p(t | w) = p(w | t) * p(t) / p(w)
    pass


def calc_terms_relevance(
        phi: Union[ndarray, DataFrame],
        topic: int,
        lambda_: float = 0.3) -> Series:
    """[summary]

    Parameters
    ----------
    phi : Union[np.ndarray, DataFrame]
        [description]
    topic : int
        [description]
    lambda_ : float, optional
        [description], by default 0.3

    Returns
    -------
    Series
        [description]
    """
    # relevance = lambda * p(w | t) + (1 - lambda) * p(w | t)/p(w)
    relevance = lambda_ * phi.loc[:, topic]\
        + (1 - lambda_) * phi.loc[:, topic] / phi.sum(axis=1)
    return relevance.sort_values(ascending=False)
