from typing import Optional, Sequence, List
from ipywidgets import widgets as wdg
from pandas import DataFrame
from _distance import get_topics_dist, get_topics_scatter
from _vis import plot_scatter_topics, plot_terms, plot_docs
from _helpers import (
    calc_terms_probs_ratio, calc_topics_marg_probs,
    get_phi, get_theta,
    get_top_docs)
from copy import deepcopy


def prepare_coords(
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
    topics_coords['size'] = (topics_marg_prob_sum
                             / topics_marg_prob_sum.sum() * 100).round(2)
    topics_coords['label'] = labels or theta.index
    return topics_coords


def report(
        model: object = None,
        topics_labels: Optional[Sequence] = None,
        vocab: Optional[Sequence] = None,
        docs: Optional[Sequence] = None,
        gensim_corpus: Optional[List] = None,
        layout: wdg.Layout = None,
        show_headers: bool = True,
        show_docs: bool = True,
        show_words: bool = True,
        show_topics: bool = True,
        topics_kws: dict = None,
        coords_kws: dict = None,
        words_kws: dict = None,
        docs_kws: dict = None,
        top_docs_kws: dict = None) -> wdg.GridBox:
    from IPython import display

    _topics_kws = {} if not topics_kws else deepcopy(topics_kws)
    _coords_kws = {} if not coords_kws else deepcopy(coords_kws)
    _words_kws = {} if not words_kws else deepcopy(words_kws)
    _top_docs_kws = {} if not docs_kws else deepcopy(top_docs_kws)
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

    if 'topics_coords' not in _topics_kws:
        topics_coords = prepare_coords(model, **_coords_kws)
        _topics_kws.update({'topics_coords': topics_coords})

    if 'terms_probs' not in _words_kws:
        phi = get_phi(model)
        terms_probs = calc_terms_probs_ratio(phi, 24)
        _words_kws.update({'terms_probs': terms_probs})

    if 'docs' not in _docs_kws:
        theta = get_theta(model, gensim_corpus=gensim_corpus).values
        _top_docs_kws.update({'docs': docs, 'theta': theta, 'topics_idx': [0]})
        top_docs = get_top_docs(**_top_docs_kws)
        _docs_kws.update({'docs': top_docs})

    topics_ids = list(range(len(_topics_kws['topics_coords'])))
    select_topic_widget = wdg.Dropdown(
        options=list(zip(topics_labels, topics_ids)),
        value=0,
        description='Select a topic:')

    # Topics scatter
    def _on_select_topics_method(names):
        topics_plot_output.clear_output(wait=False)
        with topics_plot_output:
            _coords_kws.update({'scatter_kws': {'method': names['new']}})
            topics_coords = prepare_coords(model, **_coords_kws)
            _topics_kws.update({'topics_coords': topics_coords, 'topic_id': 0})
            display(plot_scatter_topics(**_topics_kws))

    if show_topics:
        topics_plot_children = [topics_header] if show_headers else []
        options_methods = [
            ('Fruchterman-Reingold force-directed algorithm', 'graph'),
            ('TSNE', 'tsne'),
            ('SpectralEmbedding', 'sem'),
            ('MDS', 'mds'),
            ('LocallyLinearEmbedding', 'lle'),
            ('Isomap', 'isomap')
        ]
        topics_method = wdg.Dropdown(
            options=options_methods,
            value='graph',
            description='Select a method for plotting topics:',
        )
        topics_method.observe(_on_select_topics_method, names='value')
        topics_plot_output = wdg.Output()
        topics_plot = plot_scatter_topics(**_topics_kws)
        topics_plot_output.append_display_data(topics_plot)
        topics_plot_children.extend([topics_method, topics_plot_output])
        topics_widget = wdg.VBox(topics_plot_children)
        children.append(topics_widget)

    # Words
    if show_words:
        words_plot = plot_terms(**_words_kws)
        words_plot_children = [words_header] if show_headers else []
        words_plot_children.append(words_plot)
        words_widget = wdg.VBox(words_plot_children)
        children.append(words_widget)

    # Docs
    if show_docs:
        docs_plot = plot_docs(**_docs_kws)
        docs_plot_children = [docs_header] if show_headers else []
        docs_plot_children.append(docs_plot)
        docs_widget = wdg.VBox(docs_plot_children)
        children.append(docs_widget)

    grid_box = wdg.GridBox(children, layout=layout)
    app = wdg.VBox([select_topic_widget, grid_box])

    return app
