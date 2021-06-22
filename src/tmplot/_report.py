from typing import Optional, Sequence
from ipywidgets import widgets as wdg
from pandas import DataFrame
from _distance import get_topics_dist, get_topics_scatter
from _vis import plot_scatter_topics, plot_terms, plot_docs
from _helpers import (
    calc_terms_probs_ratio, calc_topics_marg_probs, get_phi, get_theta)
from copy import deepcopy


def prepare(
        model: object,
        labels: Optional[Sequence] = None,
        dist_kws: dict = None,
        scatter_kws: dict = None) -> DataFrame:
    """[summary]

    Parameters
    ----------
    model : object
        [description]
    dist_kws : dict, optional
        [description], by default None
    scatter_kws : dict, optional
        [description], by default None
    """
    if not dist_kws:
        dist_kws = {}
    if not scatter_kws:
        scatter_kws = {}

    phi = get_phi(model)
    theta = get_theta(model)
    topics_dists = get_topics_dist(phi, **dist_kws)
    topics_marg_prob_sum = calc_topics_marg_probs(theta)
    topics_coords = get_topics_scatter(topics_dists, theta, **scatter_kws)
    topics_coords['size'] = topics_marg_prob_sum
    topics_coords['labels'] = labels or theta.index
    return topics_coords


def report(
        model: object = None,
        vocab: Optional[Sequence] = None,
        docs: Optional[Sequence] = None,
        layout: wdg.Layout = None,
        show_headers: bool = True,
        show_docs: bool = True,
        show_words: bool = True,
        show_topics: bool = True,
        topics_kws: dict = None,
        words_kws: dict = None,
        docs_kws: dict = None) -> wdg.GridBox:
    _topics_kws = {} if not topics_kws else deepcopy(topics_kws)
    _words_kws = {} if not words_kws else deepcopy(words_kws)
    _docs_kws = {} if not docs_kws else deepcopy(docs_kws)

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

    if 'topics_coords' not in _topics_kws:
        topics_coords = prepare(model)
        _topics_kws.update({'topics_coords': topics_coords})

    if 'terms_probs' not in _words_kws:
        phi = get_phi(model)
        terms_probs = calc_terms_probs_ratio(phi, 24)
        _words_kws.update({'terms_probs', terms_probs})
    
    #docs_kws.update({})

    for active, header, func, kwargs in zip(show, headers, funcs, funcs_args):
        if active:
            plot = func(**funcs_args)
            plot_children = [header] if show_headers else []
            plot_children.append(plot)
            plot_widget = wdg.VBox(plot_children)
            children.append(plot_widget)

    grid_box = wdg.GridBox(children, layout=layout)

    return grid_box
