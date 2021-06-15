__all__ = [
    'get_phi', 'get_theta', 'get_relevant_terms', 'get_salient_terms',
    'calc_terms_marg_probs', 'calc_topics_marg_probs']
from typing import Union, Optional, Sequence
from functools import partial
from numpy import ndarray, zeros
from pandas import concat, Series, DataFrame
from tomotopy import (
    LDAModel as tomotopyLDA,
    LLDAModel as tomotopyLLDA)
from gensim.models.ldamodel import LdaModel as gensimLDA
from bitermplus._btm import BTM


def get_phi(
        model: object,
        vocabulary: Optional[Sequence] = None) -> DataFrame:
    """Returns topics (T) vs words (W) matrix of shape (T, W)."""

    if _is_tomotopy(model):

        # Topics vs words distributions
        twd = map(
            lambda x: Series(model.get_topic_word_dist(x)),
            range(model.k))

        # Concatenating into DataFrame
        phi = concat(twd, axis=1)

        # Specifying terms from vocabulary as index
        phi.index = model.vocabs

    elif _is_gensim(model):

        twd = DataFrame(model.get_topics().T)
        if vocabulary:
            twd.index = vocabulary

    elif _is_btmplus(model):
        twd = model.df_words_topics_
        pass

    return phi


def _is_tomotopy(model: object) -> bool:
    tomotopy_models = [tomotopyLDA, tomotopyLLDA]
    return any(map(partial(isinstance, model), tomotopy_models))


def _is_gensim(model: object) -> bool:
    gensim_models = [gensimLDA]
    return any(map(partial(isinstance, model), gensim_models))


def _is_btmplus(model: object) -> bool:
    return isinstance(model, BTM)


def get_theta(
        model: object,
        gensim_corpus: list = None) -> DataFrame:

    if _is_tomotopy(model):
        tdd = map(lambda x: Series(x.get_topic_dist()), model.docs)
        theta = concat(tdd, axis=1)

    elif _is_gensim(model):
        tdd = list(map(model.get_document_topics, gensim_corpus))
        theta = DataFrame(zeros((len(tdd), model.num_topics)))
        for doc_id, doc_topic in enumerate(tdd):
            for topic_id, topic_prob in doc_topic:
                theta.loc[doc_id, topic_id] = topic_prob

    elif _is_btmplus(model):
        theta = DataFrame(model.matrix_topics_docs_)

    return theta


def get_top_docs(
        model: object = None,
        theta: ndarray = None,
        docs: Optional[Sequence] = None) -> DataFrame:
    if model:
        pass
    elif theta:
        pass


def calc_topics_marg_probs(
        theta: Union[DataFrame, ndarray]):
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


def get_salient_terms(
        terms_freqs: ndarray,
        phi: ndarray,
        ):
    # saliency(term w) = frequency(w) * [sum_t p(t | w) * log(p(t | w)/p(t))] for topics t
    # p(t | w) = p(w | t) * p(t) / p(w)
    pass


def calc_terms_probs_ratio(
        phi: DataFrame,
        topic: Union[str, int],
        terms_num: int = 30,
        lambda_: float = 0.3):
    terms_probs = concat((
        phi.sum(axis=1).rename('Marginal term probability, p(w)'),
        phi.loc[:, topic].rename('Conditional term probability, p(w|t)')
        ), axis=1)

    relevant_idx = get_relevant_terms(phi, topic, lambda_).index
    terms_probs_slice = terms_probs.loc[relevant_idx].head(terms_num)

    return terms_probs_slice\
        .reset_index(drop=False)\
        .melt(id_vars=['index'], var_name='Type', value_name='Probability')\
        .rename(columns={'index': 'Terms'})


def get_relevant_terms(
        phi: Union[ndarray, DataFrame],
        topic: Union[str, int],
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
