__all__ = [
    'get_phi', 'get_theta',
    'get_relevant_terms', 'get_salient_terms',
    'get_docs', 'get_top_docs',
    'calc_terms_marg_probs', 'calc_topics_marg_probs',
    'calc_terms_probs_ratio']
from typing import Union, Optional, Sequence, List
from functools import partial
from math import log
from numpy import ndarray, zeros, argsort, array, arange, vstack
from pandas import concat, Series, DataFrame
from tomotopy import (
    LDAModel as tomotopyLDA,
    LLDAModel as tomotopyLLDA,
    CTModel as tomotopyCT,
    DMRModel as tomotopyDMR,
    HDPModel as tomotopyHDP,
    PTModel as tomotopyPT,
    SLDAModel as tomotopySLDA,
    GDMRModel as tomotopyGDMR)
from gensim.models.ldamodel import LdaModel as gensimLDA
from gensim.models.ldamulticore import LdaMulticore as gensimLDAMC
from bitermplus._btm import BTM


def get_phi(
        model: object,
        vocabulary: Optional[Sequence] = None) -> DataFrame:
    """Get words vs topics matrix (phi).

    Returns ``phi`` matrix of shape W x T, where W is the number of words,
    and T is the number of topics.

    Parameters
    ----------
    model : object
        Topic model instance.
    vocabulary : Optional[Sequence], optional
        Vocabulary as a list of words. Needed for getting ``phi`` matrix
        from ``gensim`` model instance.

    Returns
    -------
    DataFrame
        Words vs topics matrix (phi).
    """
    if _is_tomotopy(model):

        # Topics vs words distributions
        twd = list(map(model.get_topic_word_dist, range(model.k)))

        # Concatenating into DataFrame
        phi = DataFrame(vstack(twd).T)

        # Specifying terms from vocabulary as index
        phi.index = model.vocabs

    elif _is_gensim(model):

        phi = DataFrame(model.get_topics().T)
        if vocabulary:
            phi.index = vocabulary

    elif _is_btmplus(model):
        phi = model.df_words_topics_

    if isinstance(phi, DataFrame):
        phi.index.name = 'words'
        phi.columns.name = 'topics'

    return phi


def _is_tomotopy(model: object) -> bool:
    tomotopy_models = [
        tomotopyLDA, tomotopyLLDA, tomotopyCT, tomotopyDMR, tomotopyHDP,
        tomotopyPT, tomotopySLDA, tomotopyGDMR]
    return any(map(partial(isinstance, model), tomotopy_models))


def _is_gensim(model: object) -> bool:
    gensim_models = [gensimLDA, gensimLDAMC]
    return any(map(partial(isinstance, model), gensim_models))


def _is_btmplus(model: object) -> bool:
    return isinstance(model, BTM)


def get_theta(
        model: object,
        corpus: Optional[List] = None) -> DataFrame:
    """Get topics vs documents (theta) matrix.

    Returns theta matrix of shape T x D, where T is the number of topics,
    D is the number of documents.

    Parameters
    ----------
    model : object
        Topic model instance.
    corpus : Optional[List], optional
        Corpus.

    Returns
    -------
    DataFrame
        Topics vs documents matrix (theta).
    """
    theta = None

    if _is_tomotopy(model):
        tdd = list(map(lambda x: x.get_topic_dist(), model.docs))
        theta = DataFrame(vstack(tdd).T)

    elif _is_gensim(model):
        if corpus is None:
            raise ValueError(
                '`corpus` must be supplied for a gensim model')
        tdd = list(map(model.get_document_topics, corpus))
        theta = DataFrame(zeros((len(tdd), model.num_topics)))
        for doc_id, doc_topic in enumerate(tdd):
            for topic_id, topic_prob in doc_topic:
                theta.loc[doc_id, topic_id] = topic_prob
        theta = theta.T

    elif _is_btmplus(model):
        theta = DataFrame(model.matrix_topics_docs_)

    if isinstance(theta, DataFrame):
        theta.index.name = 'topics'
        theta.columns.name = 'docs'

    return theta


def get_docs(
        model: object) -> List[str]:
    """Retrieve documents from topic model object.

    Parameters
    ----------
    model : object
        Topic model instance.

    Returns
    -------
    List[str]
        List of documents.
    """
    if _is_tomotopy(model):
        docs_raw = map(lambda x: x.words, model.docs)
        return list(
            map(
                lambda doc: " ".join(map(lambda x: model.vocabs[x], doc)),
                docs_raw))
    return None


def get_top_docs(
        docs: Sequence[str],
        model: object = None,
        theta: ndarray = None,
        corpus: Optional[List] = None,
        docs_num: int = 5,
        topics: Sequence[int] = None) -> DataFrame:
    """Get top documents for all (or a selected) topic.

    Parameters
    ----------
    docs : Sequence
        List of documents.
    model : object, optional
        Topic model instance.
    theta : ndarray, optional
        Topics vs documents matrix.
    corpus : Optional[List], optional
        Corpus for ``gensim`` model.
    docs_num : int, optional
        Number of documents to return.
    topics : Sequence[int], optional
        Sequence of topics indices.

    Returns
    -------
    DataFrame
        Top documents.

    Raises
    ------
    ValueError
        If neither a model or theta matrix is passed, ValueError is raised.
    """
    if all([model is None, theta is None]):
        raise ValueError("Please pass a model or a theta matrix to function")

    if model and not theta:
        theta = get_theta(model, corpus=corpus).values

    def _select_docs(docs, theta, topic_id: int):
        probs = theta[topic_id, :]
        idx = argsort(probs)[:-docs_num-1:-1]
        result = Series(list(map(lambda x: docs[x], idx)))
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = theta.shape[0]
    topics_idx = arange(topics_num) if topics is None else topics
    return concat(
        map(lambda x: _select_docs(docs, theta, x), topics_idx), axis=1)


def calc_topics_marg_probs(
        theta: Union[DataFrame, ndarray],
        topic_id: int = None) -> Union[DataFrame, ndarray]:
    """Calculate marginal topics probabilities.

    Parameters
    ----------
    theta : Union[DataFrame, ndarray]
        Topics vs documents matrix.
    topic_id : int, optional
        Topic index.

    Returns
    -------
    Union[DataFrame, ndarray]
        Marginal topics probabilities.
    """
    if topic_id:
        if isinstance(theta, ndarray):
            return theta[topic_id, :].sum()
        elif isinstance(theta, DataFrame):
            return theta.iloc[topic_id, :].sum()
    else:
        return theta.sum(axis=1)


def calc_terms_marg_probs(
        phi: Union[ndarray, DataFrame],
        word_id: Optional[int] = None) -> Union[ndarray, Series]:
    """Calculate marginal terms probabilities.

    Parameters
    ----------
    phi : Union[ndarray, DataFrame]
        Words vs topics matrix.
    word_id: Optional[int]
        Word index.

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
    """Get salient terms.

    Calculated as:
    saliency(w) = frequency(w) * [sum_t p(t | w) * log(p(t | w)/p(t))],
    where ``w`` is a term index, ``t`` is a topic index.

    Parameters
    ----------
    terms_freqs : numpy.ndarray
        Words frequencies.
    phi : numpy.ndarray
        Words vs topics matrix.
    theta : numpy.ndarray
        Topics vs documents matrix.

    Returns
    -------
    numpy.ndarray
        Terms saliency values.
    """
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
        lambda_: float = 0.6) -> DataFrame:
    """Get terms conditional and marginal probabilities.

    Parameters
    ----------
    phi : DataFrame
        Words vs topics matrix.
    topic : int
        Topic index.
    terms_num : int, optional
        Number of words to return.
    lambda_ : float, optional
        Weight parameter. It determines the weight given to the probability
        of term W under topic T relative to its lift [1]_. Setting it to 1
        equals topic-specific probabilities of terms.

    References
    ----------
    .. [1] Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing
           and interpreting topics. In Proceedings of the workshop on
           interactive language learning, visualization, and interfaces (pp.
           63-70).

    Returns
    -------
    DataFrame
        Words conditional and marginal probabilities.
    """
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
        .melt(
            id_vars=[terms_probs_slice.index.name],
            var_name='Type',
            value_name='Probability')\
        .rename(columns={terms_probs_slice.index.name: 'Terms'})


def get_relevant_terms(
        phi: Union[ndarray, DataFrame],
        topic: int,
        lambda_: float = 0.6) -> Series:
    """Select relevant terms.

    Parameters
    ----------
    phi : Union[np.ndarray, DataFrame]
        Words vs topics matrix (phi).
    topic : int
        Topic index.
    lambda_ : float = 0.6
        Weight parameter. It determines the weight given to the probability
        of term W under topic T relative to its lift [1]_. Setting it to 1
        equals topic-specific probabilities of terms.

    References
    ----------
    .. [1] Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing
           and interpreting topics. In Proceedings of the workshop on
           interactive language learning, visualization, and interfaces (pp.
           63-70).

    Returns
    -------
    Series
        Terms sorted by relevance (descendingly).
    """
    phi_topic = phi.iloc[:, topic]\
        if isinstance(phi, DataFrame)\
        else phi[:, topic]
    # relevance = lambda * p(w | t) + (1 - lambda) * p(w | t)/p(w)
    relevance = lambda_ * phi_topic\
        + (1 - lambda_) * phi_topic / phi.sum(axis=1)
    return relevance.sort_values(ascending=False)
