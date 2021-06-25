__all__ = [
    'get_phi', 'get_theta',
    'get_relevant_terms', 'get_salient_terms',
    'get_docs', 'get_top_docs',
    'calc_terms_marg_probs', 'calc_topics_marg_probs']
from typing import Union, Optional, Sequence, List
from functools import partial
from math import log
from numpy import ndarray, zeros, argsort, array, arange, vstack
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
        twd = list(map(lambda x: model.get_topic_word_dist(x), range(model.k)))

        # Concatenating into DataFrame
        phi = DataFrame(vstack(twd).T)

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
        gensim_corpus: Optional[List] = None) -> DataFrame:

    if _is_tomotopy(model):
        tdd = list(map(lambda x: x.get_topic_dist(), model.docs))
        theta = DataFrame(vstack(tdd).T)

    elif _is_gensim(model):
        tdd = list(map(model.get_document_topics, gensim_corpus))
        theta = DataFrame(zeros((len(tdd), model.num_topics)))
        for doc_id, doc_topic in enumerate(tdd):
            for topic_id, topic_prob in doc_topic:
                theta.loc[doc_id, topic_id] = topic_prob

    elif _is_btmplus(model):
        theta = DataFrame(model.matrix_topics_docs_)

    return theta


def get_docs(
        model: object) -> List[str]:
    if _is_tomotopy(model):
        docs_raw = map(lambda x: x.words, model.docs)
        return list(
            map(
                lambda doc: " ".join(map(lambda x: model.vocabs[x], doc)),
                docs_raw))
    return None


def get_top_docs(
        docs: Sequence,
        model: object = None,
        theta: ndarray = None,
        gensim_corpus: Optional[List] = None,
        docs_num: int = 20,
        topics_idx: Sequence[int] = None) -> DataFrame:
    if all([model is None, theta is None]):
        raise ValueError("Please pass a model or a theta matrix to function")

    if model and not theta:
        theta = get_theta(model, gensim_corpus=gensim_corpus).values

    def _select_docs(docs, theta, topic_id: int):
        probs = theta[topic_id, :]
        idx = argsort(probs)[:-docs_num-1:-1]
        result = Series(list(map(lambda x: docs[x], idx)))
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = theta.shape[0]
    topics_idx = arange(topics_num) if topics_idx is None else topics_idx
    return concat(
        map(lambda x: _select_docs(docs, theta, x), topics_idx), axis=1)


def calc_topics_marg_probs(
        theta: Union[DataFrame, ndarray],
        topic_id: int = None):
    """Calculate marginal topics probabilities"""
    return theta[topic_id, :].sum() if topic_id else theta.sum(axis=1)


def calc_terms_marg_probs(
        phi: Union[ndarray, DataFrame],
        word_id: Optional[int] = None) -> Union[ndarray, Series]:
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
    if word_id:
        if isinstance(phi, ndarray):
            return phi[word_id, :].sum()
        elif isinstance(phi, DataFrame):
            return phi.iloc[word_id, :].sum()
    else:
        return phi.sum(axis=1)


def get_salient_terms(
        terms_freqs: ndarray,
        phi: ndarray,
        theta: ndarray) -> ndarray:
    p_t = array(calc_topics_marg_probs(theta))
    p_w = array(calc_terms_marg_probs(phi))

    def _p_tw(phi, w, t):
        return phi[w, t] * p_t[t] / p_w[w]

    saliency = array([
        terms_freqs[w] * sum([
            _p_tw(phi, w, t) * log(_p_tw(phi, w, t) / p_t[t])
            for t in range(phi.shape[1])])
        for w in range(phi.shape[0])
    ])
    # saliency(term w) = frequency(w)
    # * [sum_t p(t | w) * log(p(t | w)/p(t))] for topics t
    # p(t | w) = p(w | t) * p(t) / p(w)
    return saliency


def calc_terms_probs_ratio(
        phi: DataFrame,
        topic: int,
        terms_num: int = 30,
        lambda_: float = 0.3):
    p_cond_name = 'Conditional term probability, p(w|t)'
    p_cond = phi.iloc[:, topic]\
        .rename(p_cond_name)\
        if isinstance(phi, DataFrame)\
        else Series(phi[:, topic], name=p_cond_name)

    p_marg_name = 'Marginal term probability, p(w)'
    p_marg = phi.sum(axis=1)\
        .rename(p_marg_name)\
        if isinstance(phi, DataFrame)\
        else Series(phi[:, topic], name=p_marg_name)

    terms_probs = concat((p_marg, p_cond), axis=1)
    relevant_idx = get_relevant_terms(phi, topic, lambda_).index
    terms_probs_slice = terms_probs.loc[relevant_idx].head(terms_num)

    return terms_probs_slice\
        .reset_index(drop=False)\
        .melt(id_vars=['index'], var_name='Type', value_name='Probability')\
        .rename(columns={'index': 'Terms'})


def get_relevant_terms(
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
    phi_topic = phi.iloc[:, topic]\
        if isinstance(phi, DataFrame)\
        else phi[:, topic]
    # relevance = lambda * p(w | t) + (1 - lambda) * p(w | t)/p(w)
    relevance = lambda_ * phi_topic\
        + (1 - lambda_) * phi_topic / phi.sum(axis=1)
    return relevance.sort_values(ascending=False)
