from typing import Iterable, Optional
from ipywidgets import widgets as wdg
from numpy import ndarray
from pandas import DataFrame
from ._vis import plot_scatter_topics, plot_terms, plot_docs
from ._distance import get_topics_dist, get_topics_scatter
from ._helpers import calc_topics_marg_probs


def prepare(
        phi: ndarray,
        theta: ndarray,
        labels: Optional[Iterable] = None,
        dist_kws: dict = None,
        scatter_kws: dict = None) -> DataFrame:
    """[summary]

    Parameters
    ----------
    phi : ndarray
        [description]
    theta : ndarray
        [description]
    dist_kws : dict, optional
        [description], by default None
    scatter_kws : dict, optional
        [description], by default None
    """
    topics_dists = get_topics_dist(phi, **dist_kws)
    topics_marg_prob_sum = calc_topics_marg_probs(theta)
    topics_coords = get_topics_scatter(topics_dists, theta, **scatter_kws)
    topics_coords['size'] = topics_marg_prob_sum
    topics_coords['labels'] = labels or theta.columns
    return topics_coords


def report(
        model: object = None,
        theta: ndarray = None,
        phi: ndarray = None,
        vocab: Iterable = None,
        docs: Iterable = None,
        layout: wdg.Layout = None,
        show_headers: bool = True,
        show_docs: bool = True,
        show_words: bool = True,
        show_topics: bool = True,
        topics_kws: dict = None,
        words_kws: dict = None,
        docs_kws: dict = None) -> wdg.GridBox:
    # Headers init
    topics_header = wdg.HTML('<b>Topics scatter plot</b>')\
        if show_headers and show_topics else None
    words_header = wdg.HTML('<b>Relevant words (terms)</b>')\
        if show_headers and show_words else None
    docs_header = wdg.HTML('<b>Top documents in a topic</b>')\
        if show_headers and show_docs else None

    # Layout init
    grid_cols = " ".join(['1fr'] * sum([show_docs, show_words, show_topics]))
    layout = wdg.Layout(
            grid_template_columns=grid_cols,
            justify_items='center')\
        if not layout else layout

    # Children widgets list init
    children = []
    headers = [topics_header, words_header, docs_header]
    funcs = [plot_scatter_topics, plot_terms, plot_docs]
    funcs_args = [topics_kws, words_kws, docs_kws]
    show = [show_topics, show_words, show_docs]

    topics_kws.update({})
    words_kws.update({})
    docs_kws.update({})

    for active, header, func, kwargs in zip(show, headers, funcs, funcs_args):
        if active:
            plot = func(**funcs_args)
            plot_children = [header] if show_headers else []
            plot_children.append(plot)
            plot_widget = wdg.VBox(plot_children)
            children.append(plot_widget)

    grid_box = wdg.GridBox(children, layout=layout)

    return grid_box
