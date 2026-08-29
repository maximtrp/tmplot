from __future__ import annotations

__all__ = [
    "calc_terms_marg_probs",
    "calc_terms_probs_ratio",
    "calc_topics_marg_probs",
    "get_docs",
    "get_phi",
    "get_relevant_terms",
    "get_salient_terms",
    "get_theta",
    "get_top_docs",
]
from collections.abc import Sequence
from functools import partial
from importlib.util import find_spec
from typing import Optional, Union
from warnings import warn

import numpy as np
from numpy import arange, array, ndarray, vstack, zeros
from numpy import log as nplog
from pandas import DataFrame, Series, concat

tomotopy_installed = find_spec("tomotopy")
if tomotopy_installed:
    try:
        from tomotopy import (
            CTModel as tomotopyCT,
        )
        from tomotopy import (
            DMRModel as tomotopyDMR,
        )
        from tomotopy import (
            GDMRModel as tomotopyGDMR,
        )
        from tomotopy import (
            HDPModel as tomotopyHDP,
        )
        from tomotopy import (
            LDAModel as tomotopyLDA,
        )
        from tomotopy import (
            LLDAModel as tomotopyLLDA,
        )
        from tomotopy import (
            PTModel as tomotopyPT,
        )
        from tomotopy import (
            SLDAModel as tomotopySLDA,
        )
    except (ImportError, OSError):
        tomotopy_installed = None

gensim_installed = find_spec("gensim")
if gensim_installed:
    try:
        from gensim.models.ldamodel import LdaModel as gensimLDA
        from gensim.models.ldamulticore import LdaMulticore as gensimLDAMC
    except (ImportError, OSError):
        gensim_installed = None

bitermplus_installed = find_spec("bitermplus")
if bitermplus_installed:
    try:
        from bitermplus import BTMClassifier
        from bitermplus._btm import BTM
    except (ImportError, OSError):
        bitermplus_installed = None


def _warn_missing_model_packages() -> None:
    missing = [
        name
        for name, installed in (
            ("tomotopy", tomotopy_installed),
            ("gensim", gensim_installed),
            ("bitermplus", bitermplus_installed),
        )
        if not installed
    ]
    if missing:
        packages = ", ".join(missing)
        warn(
            f"Optional model adapter packages are not installed: {packages}. "
            f"Install the required adapter to analyze its models.",
            UserWarning,
            stacklevel=2,
        )


def _live_topic_ids(model: object) -> list[int]:
    """Topic ids worth exposing for a fitted ``tomotopy`` model.

    Nonparametric models (``HDPModel``) allocate topics that die out during
    sampling. ``model.k`` keeps counting them, but their word distributions are
    uniform placeholders carrying no document mass, so they would otherwise
    appear as spurious topics in every downstream analysis - distance matrices,
    scatter plots and, most damagingly, entropy-based estimates of the optimal
    number of topics.
    """
    if not hasattr(model, "is_live_topic"):
        return list(range(model.k))

    live = [topic for topic in range(model.k) if model.is_live_topic(topic)]
    if not live:
        raise ValueError(
            f"This {type(model).__name__} has no live topics: all {model.k} "
            "allocated topics died out during training. The model is probably "
            "undertrained, or the corpus too small to support any topic."
        )
    dropped = model.k - len(live)
    if dropped:
        warn(
            f"Dropped {dropped} dead topic(s) out of {model.k} from this "
            f"{type(model).__name__}; {len(live)} live topics remain. Topic ids "
            "are preserved as the phi columns / theta index.",
            UserWarning,
            stacklevel=3,
        )
    return live


def get_phi(model: object, vocabulary: Optional[Sequence] = None) -> DataFrame:
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
    pandas.DataFrame
        Words vs topics matrix (phi).
    """
    phi = None

    if _is_tomotopy(model):
        # Topics vs words distributions, skipping topics that died out
        topic_ids = _live_topic_ids(model)
        twd = list(map(model.get_topic_word_dist, topic_ids))

        # Concatenating into DataFrame, keeping the model's own topic ids
        phi = DataFrame(vstack(twd).T, columns=topic_ids)

        # Specifying terms from vocabulary as index
        phi.index = list(model.used_vocabs)

    elif _is_gensim(model):
        phi = DataFrame(model.get_topics().T)
        if vocabulary is not None:
            if len(vocabulary) != phi.shape[0]:
                raise ValueError("vocabulary length must match the number of words")
            phi.index = vocabulary

    elif _is_btm_classifier(model):
        # The wrapped BTM exposes phi as a words x topics frame already
        # indexed by the vocabulary.
        phi = model.model_.df_words_topics_

    elif _is_btmplus(model):
        phi = model.df_words_topics_

    if isinstance(phi, DataFrame):
        phi.index.name = "words"
        phi.columns.name = "topics"

    if phi is None:
        _warn_missing_model_packages()
        raise ValueError(f"Unsupported model type: {type(model)}")
    return phi


def _is_tomotopy(model: object) -> bool:
    if tomotopy_installed:
        tomotopy_models = [
            tomotopyLDA,
            tomotopyLLDA,
            tomotopyCT,
            tomotopyDMR,
            tomotopyHDP,
            tomotopyPT,
            tomotopySLDA,
            tomotopyGDMR,
        ]
        return any(map(partial(isinstance, model), tomotopy_models))

    return False


def _is_gensim(model: object) -> bool:
    if gensim_installed:
        gensim_models = [gensimLDA, gensimLDAMC]
        return any(map(partial(isinstance, model), gensim_models))

    return False


def _is_btmplus(model: object) -> bool:
    if bitermplus_installed:
        return isinstance(model, (BTM, BTMClassifier))

    return False


def _is_btm_classifier(model: object) -> bool:
    """``BTMClassifier`` keeps document-topic state that plain ``BTM`` does not."""
    return bool(bitermplus_installed and isinstance(model, BTMClassifier))


def get_theta(model: object, corpus: Optional[list] = None) -> Optional[DataFrame]:
    """Get topics vs documents (theta) matrix.

    Returns theta matrix of shape T x D, where T is the number of topics,
    D is the number of documents.

    Parameters
    ----------
    model : object
        Topic model instance.
    corpus : Optional[List], optional
        Corpus. Required for a `gensim` model (bag-of-words corpus) and for a
        plain `bitermplus` ``BTM`` model (vectorized documents from
        ``bitermplus.get_vectorized_docs()``). Not needed for ``tomotopy``
        models or for ``BTMClassifier``.

    Returns
    -------
    pandas.DataFrame
        Topics vs documents matrix (theta).
    """
    theta = None

    if _is_tomotopy(model):
        # get_topic_dist() spans all model.k topics; keep only the live ones so
        # that theta stays aligned with get_phi.
        topic_ids = _live_topic_ids(model)
        tdd = [x.get_topic_dist() for x in model.docs]
        theta = DataFrame(vstack(tdd).T).iloc[topic_ids]

    elif _is_gensim(model):
        if corpus is None:
            raise ValueError("`corpus` must be supplied for a gensim model")
        if len(corpus) == 0:
            raise ValueError("corpus cannot be empty")
        tdd = list(map(model.get_document_topics, corpus))
        theta_values = zeros((len(tdd), model.num_topics))
        for doc_id, doc_topic in enumerate(tdd):
            for topic_id, topic_prob in doc_topic:
                theta_values[doc_id, topic_id] = topic_prob
        theta = DataFrame(theta_values.T)

    elif _is_btm_classifier(model):
        # Fitted over the training documents, so no corpus is needed.
        theta = DataFrame(model.matrix_docs_topics_).T

    elif _is_btmplus(model):
        if corpus is None:
            raise ValueError(
                "`corpus` must be supplied for a bitermplus BTM model. Since "
                "bitermplus 1.0 the document-topic matrix is not stored on the "
                "model, because p(z|d) is inferred per document rather than "
                "fitted. Pass the vectorized documents returned by "
                "bitermplus.get_vectorized_docs(), or use BTMClassifier, which "
                "keeps matrix_docs_topics_ for its training documents."
            )
        theta = DataFrame(model.transform(corpus, verbose=False)).T
    else:
        _warn_missing_model_packages()
        raise ValueError(f"Unsupported model type: {type(model)}")

    if isinstance(theta, DataFrame):
        theta.index.name = "topics"
        theta.columns.name = "docs"

    return theta


def get_docs(model: object) -> Optional[list[str]]:
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
        docs_raw = (x.words for x in model.docs)
        return [" ".join(model.vocabs[x] for x in doc) for doc in docs_raw]
    return None


def get_top_docs(
    docs: Sequence[str],
    model: object = None,
    theta: Optional[ndarray] = None,
    corpus: Optional[list] = None,
    docs_num: int = 5,
    topics: Optional[Sequence[int]] = None,
) -> DataFrame:
    """Get top documents for all (or a selected) topic.

    Parameters
    ----------
    docs : Sequence
        List of documents.
    model : object, optional
        Topic model instance.
    theta : numpy.ndarray, optional
        Topics vs documents matrix.
    corpus : Optional[List], optional
        Corpus for ``gensim`` model.
    docs_num : int, optional
        Number of documents to return.
    topics : Sequence[int], optional
        Sequence of topics indices.

    Returns
    -------
    pandas.DataFrame
        Top documents.

    Raises
    ------
    ValueError
        If neither a model or theta matrix is passed, ValueError is raised.
    """
    if all([model is None, theta is None]):
        raise ValueError("Please pass a model or a theta matrix to function")

    if theta is None:
        theta = get_theta(model, corpus=corpus).to_numpy()

    theta = array(theta)
    if theta.ndim != 2:
        raise ValueError("theta must be a 2D topics x documents matrix")
    if len(docs) != theta.shape[1]:
        raise ValueError("docs length must match theta documents dimension")
    if docs_num <= 0:
        raise ValueError("docs_num must be positive")

    def _select_docs(docs, theta, topic_id: int):
        probs = theta[topic_id, :]
        count = min(docs_num, probs.size)
        idx = np.argpartition(probs, -count)[-count:]
        idx = idx[np.argsort(probs[idx])[::-1]]
        result = Series([docs[x] for x in idx])
        result.name = f"topic{topic_id}"
        return result

    topics_num = theta.shape[0]
    if topics is None:
        topics_idx = arange(topics_num)
    else:
        topics_idx = array(topics)
        out_of_range = topics_idx[(topics_idx < 0) | (topics_idx >= topics_num)]
        if out_of_range.size:
            raise IndexError(
                f"topics contains indices outside [0, {topics_num - 1}]: "
                f"{sorted(set(out_of_range.tolist()))}"
            )
    return concat((_select_docs(docs, theta, x) for x in topics_idx), axis=1)


def calc_topics_marg_probs(
    theta: Union[DataFrame, ndarray], topic_id: Optional[int] = None
) -> ndarray:
    """Calculate marginal topics probabilities.

    Parameters
    ----------
    theta : Union[pandas.DataFrame, numpy.ndarray]
        Topics vs documents matrix.
    topic_id : int, optional
        Topic index.

    Returns
    -------
    Union[pandas.DataFrame, numpy.ndarray]
        Marginal topics probabilities.
    """
    theta_arr = array(theta)
    if theta_arr.size == 0:
        raise ValueError("theta matrix cannot be empty")
    if theta_arr.ndim != 2:
        raise ValueError("theta matrix must be a 2D array")
    if not np.isfinite(theta_arr).all() or np.any(theta_arr < 0):
        raise ValueError("theta matrix must contain finite non-negative values")

    p_t = theta_arr.sum(axis=1)
    total_sum = p_t.sum()
    if total_sum <= 0:
        raise ValueError("theta matrix contains all zeros - cannot normalize")

    p_t /= total_sum
    if topic_id is not None:
        if topic_id < 0 or topic_id >= len(p_t):
            raise IndexError(f"topic_id {topic_id} out of bounds for {len(p_t)} topics")
        return p_t[topic_id]
    return p_t


def calc_terms_marg_probs(
    phi: Union[ndarray, DataFrame],
    p_t: Union[ndarray, Series],
    word_id: Optional[int] = None,
) -> ndarray:
    """Calculate marginal terms probabilities.

    Parameters
    ----------
    phi : Union[numpy.ndarray, pandas.DataFrame]
        Words vs topics matrix.
    p_t : Union[numpy.ndarray, pandas.Series]
        Topic marginal probabilities.
    word_id: Optional[int]
        Word index.

    Returns
    -------
    Union[numpy.ndarray, pandas.Series]
        Marginal terms probabilities.
    """
    phi_arr = array(phi)
    p_t_arr = array(p_t)

    if phi_arr.size == 0:
        raise ValueError("phi matrix cannot be empty")
    if phi_arr.ndim != 2:
        raise ValueError("phi matrix must be a 2D array")
    if p_t_arr.size == 0:
        raise ValueError("p_t array cannot be empty")
    if p_t_arr.ndim != 1:
        raise ValueError("p_t array must be a 1D array")
    if not np.isfinite(phi_arr).all() or np.any(phi_arr < 0):
        raise ValueError("phi matrix must contain finite non-negative values")
    if not np.isfinite(p_t_arr).all() or np.any(p_t_arr < 0):
        raise ValueError("p_t must contain finite non-negative values")
    if phi_arr.shape[1] != p_t_arr.shape[0]:
        raise ValueError(
            f"phi topics dimension {phi_arr.shape[1]} must match p_t length {p_t_arr.shape[0]}"
        )

    p_t_sum = p_t_arr.sum()
    if p_t_sum <= 0:
        raise ValueError("p_t must have positive total probability")
    p_w = (phi_arr * (p_t_arr / p_t_sum)).sum(axis=1)
    if word_id is not None:
        if word_id < 0 or word_id >= len(p_w):
            raise IndexError(f"word_id {word_id} out of bounds for {len(p_w)} words")
        return p_w[word_id]
    return p_w


def get_salient_terms(phi: ndarray, theta: ndarray) -> ndarray:
    """Get salient terms.

    Calculated as:
    saliency(w) = frequency(w) * [sum_t p(t | w) * log(p(t | w)/p(t))],
    where ``w`` is a term index, ``t`` is a topic index.

    Parameters
    ----------
    phi : numpy.ndarray
        Words vs topics matrix.
    theta : numpy.ndarray
        Topics vs documents matrix.

    Returns
    -------
    numpy.ndarray
        Terms saliency values.
    """
    phi = array(phi, dtype=float)
    theta = array(theta, dtype=float)
    if phi.size == 0 or theta.size == 0:
        raise ValueError("phi and theta matrices cannot be empty")
    if phi.shape[1] != theta.shape[0]:
        raise ValueError(
            f"phi topics dimension {phi.shape[1]} must match theta topics dimension {theta.shape[0]}"
        )

    p_t = calc_topics_marg_probs(theta)
    p_w = calc_terms_marg_probs(phi, p_t)

    p_tw = np.divide(
        phi * p_t,
        p_w[:, None],
        out=np.zeros_like(phi, dtype=float),
        where=p_w[:, None] > 0,
    )
    ratio = np.divide(
        p_tw,
        p_t,
        out=np.ones_like(p_tw),
        where=p_t > 0,
    )
    contributions = np.zeros_like(p_tw)
    positive = p_tw > 0
    contributions[positive] = p_tw[positive] * np.log(ratio[positive])
    return p_w * contributions.sum(axis=1)
    # saliency(term w) = frequency(w)
    # * [sum_t p(t | w) * log(p(t | w)/p(t))] for topics t
    # p(t | w) = p(w | t) * p(t) / p(w)


def _calc_relevance(
    phi: Union[ndarray, DataFrame],
    topic: int,
    lambda_: float = 0.6,
    p_t: Optional[ndarray] = None,
) -> ndarray:
    """Relevance of every term for ``topic``, in the row order of ``phi``."""
    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between 0 and 1")
    phi_arr = array(phi, dtype=float)
    if phi_arr.ndim != 2:
        raise ValueError("phi must be a 2D words x topics matrix")
    if not 0 <= topic < phi_arr.shape[1]:
        raise IndexError("topic is out of bounds")
    topic_probs = (
        np.full(phi_arr.shape[1], 1 / phi_arr.shape[1])
        if p_t is None
        else array(p_t, dtype=float)
    )
    if topic_probs.ndim != 1 or topic_probs.shape[0] != phi_arr.shape[1]:
        raise ValueError("p_t length must match the number of topics")
    p_marg = calc_terms_marg_probs(phi_arr, topic_probs)
    phi_topic = phi_arr[:, topic]

    # relevance = lambda * log(p(w | t)) + (1 - lambda) * log(p(w | t) / p(w))
    with np.errstate(divide="ignore", invalid="ignore"):
        return lambda_ * nplog(phi_topic) + (1 - lambda_) * nplog(
            np.divide(phi_topic, p_marg, out=np.zeros_like(phi_topic), where=p_marg > 0)
        )


def calc_terms_probs_ratio(
    phi: DataFrame,
    topic: int,
    terms_num: int = 30,
    lambda_: float = 0.6,
    p_t: Optional[ndarray] = None,
) -> DataFrame:
    """Get terms conditional and marginal probabilities.

    Parameters
    ----------
    phi : pandas.DataFrame
        Words vs topics matrix.
    topic : int
        Topic index.
    terms_num : int, optional
        Number of words to return.
    lambda_ : float, optional
        Weight parameter. It determines the weight given to the probability
        of term W under topic T relative to its lift [1]_. Setting it to 1
        equals topic-specific probabilities of terms.
    p_t : Union[numpy.ndarray, pandas.Series], optional
        Marginal topic probabilities. Calculated from `phi` when omitted.

    References
    ----------
    .. [1] Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing
           and interpreting topics. In Proceedings of the workshop on
           interactive language learning, visualization, and interfaces (pp.
           63-70).

    Returns
    -------
    pandas.DataFrame
        Words conditional and marginal probabilities.
    """
    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between 0 and 1")
    p_cond_name = "Conditional term probability, p(w | t)"
    p_cond = (
        phi.iloc[:, topic].rename(p_cond_name)
        if isinstance(phi, DataFrame)
        else Series(phi[:, topic], name=p_cond_name)
    )

    p_marg_name = "Marginal term probability, p(w)"
    topic_probs = np.full(phi.shape[1], 1 / phi.shape[1]) if p_t is None else array(p_t)
    marginal = calc_terms_marg_probs(phi, topic_probs)
    index = phi.index if isinstance(phi, DataFrame) else None
    p_marg = Series(marginal, index=index, name=p_marg_name)

    terms_probs = concat((p_marg, p_cond), axis=1)
    relevance = _calc_relevance(phi, topic, lambda_, p_t=p_t)
    # Rank positionally: a duplicated word label would make a .loc lookup on the
    # sorted labels expand into a cross product and crowd out relevant terms.
    order = Series(relevance).sort_values(ascending=False).index.to_numpy()
    terms_probs_slice = terms_probs.iloc[order[:terms_num]]

    return (
        terms_probs_slice.rename_axis("Terms")
        .reset_index(drop=False)
        .melt(
            id_vars=["Terms"],
            var_name="Type",
            value_name="Probability",
        )
    )


def get_relevant_terms(
    phi: Union[ndarray, DataFrame],
    topic: int,
    lambda_: float = 0.6,
    p_t: Optional[ndarray] = None,
) -> Series:
    """Select relevant terms.

    Parameters
    ----------
    phi : Union[numpy.ndarray, pandas.DataFrame]
        Words vs topics matrix (phi).
    topic : int
        Topic index.
    lambda_ : float = 0.6
        Weight parameter. It determines the weight given to the probability
        of term W under topic T relative to its lift [2]_. Setting it to 1
        equals topic-specific probabilities of terms.
    p_t : Union[numpy.ndarray, pandas.Series], optional
        Marginal topic probabilities. Calculated from `phi` when omitted.

    References
    ----------
    .. [2] Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing
           and interpreting topics. In Proceedings of the workshop on
           interactive language learning, visualization, and interfaces (pp.
           63-70).

    Returns
    -------
    pandas.Series
        Terms sorted by relevance (descendingly).
    """
    relevance = _calc_relevance(phi, topic, lambda_, p_t=p_t)
    relevance = Series(
        relevance,
        index=phi.index if isinstance(phi, DataFrame) else None,
    )
    return relevance.sort_values(ascending=False)
