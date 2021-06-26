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
        docs: Optional[Sequence] = None,
        topics_labels: Optional[Sequence] = None,
        vocab: Optional[Sequence] = None,
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
    from IPython.display import display

    _topics_kws = {'chart_kws': {'height': 600, 'width': 350}}\
        if not topics_kws else deepcopy(topics_kws)
    _coords_kws = {} if not coords_kws else deepcopy(coords_kws)
    _words_kws = {'chart_kws': {'height': 600, 'width': 250}}\
        if not words_kws else deepcopy(words_kws)
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
            # justify_items='center'
        )\
        if not layout else layout

    # Children widgets list init
    children = []

    if 'topics_coords' not in _topics_kws:
        topics_coords = prepare_coords(model, **_coords_kws)
        _topics_kws.update({
            'topics_coords': topics_coords,
            'label_col': 'label',
            'size_col': 'size'
        })

    if 'terms_probs' not in _words_kws:
        phi = get_phi(model)
        terms_probs = calc_terms_probs_ratio(phi, topic=0)
        _words_kws.update({'terms_probs': terms_probs})

    if 'docs' not in _docs_kws:
        theta = get_theta(model, gensim_corpus=gensim_corpus).values
        _top_docs_kws.update({
            'docs': docs, 'theta': theta,
            'topics': [0], 'docs_num': 2})
        top_docs = get_top_docs(**_top_docs_kws)
        _docs_kws.update({'docs': top_docs})

    # Topic selection
    def _on_select_topic(sel):
        topic = sel['new']
        with words_plot_output:
            terms_probs = calc_terms_probs_ratio(phi, topic=topic)
            _words_kws.update({'terms_probs': terms_probs})
            display(plot_terms(**_words_kws))
        with topics_plot_output:
            _topics_kws.update({'topic': topic})
            display(plot_scatter_topics(**_topics_kws))
        with docs_plot_output:
            _top_docs_kws.update({'topics': [sel['new']]})
            top_docs = get_top_docs(**_top_docs_kws)
            _docs_kws.update({'docs': top_docs})
            display(plot_docs(**_docs_kws))

    topics_ids = list(range(len(_topics_kws['topics_coords'])))
    topics_labels = topics_labels or topics_ids
    select_topic = wdg.Dropdown(
        options=list(zip(topics_labels, topics_ids)), value=0)
    select_topic.observe(_on_select_topic, names='value')
    select_topic_header = wdg.HTML('Select a topic:')
    select_topic_widget = wdg.HBox([select_topic_header, select_topic])

    # Topics scatter
    def _on_select_topics_method(names):
        topics_plot_output.clear_output(wait=False)
        with topics_plot_output:
            _coords_kws.update({'scatter_kws': {'method': names['new']}})
            topics_coords = prepare_coords(model, **_coords_kws)
            _topics_kws.update({
                'topics_coords': topics_coords,
                'topic': select_topic.value
            })
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
        topics_method_header = wdg.HTML('Select a method:')
        topics_method = wdg.Dropdown(
            options=options_methods,
            value='graph',
        )
        topics_method_widget = wdg.HBox([topics_method_header, topics_method])
        topics_method.observe(_on_select_topics_method, names='value')
        topics_plot_output = wdg.Output()
        topics_plot = plot_scatter_topics(**_topics_kws)
        topics_plot_output.append_display_data(topics_plot)
        topics_plot_children.extend([topics_method_widget, topics_plot_output])
        topics_widget = wdg.VBox(topics_plot_children)
        children.append(topics_widget)

    # Words
    if show_words:
        def _on_select_lambda(sel):
            topic = select_topic.value
            lambda_ = sel['new']
            words_plot_output.clear_output(wait=False)
            with words_plot_output:
                terms_probs = calc_terms_probs_ratio(
                    phi, topic=topic, lambda_=lambda_)
                _words_kws.update({'terms_probs': terms_probs})
                display(plot_terms(**_words_kws))

        lambda_slider = wdg.FloatSlider(
            value=0.3,
            min=0,
            max=1.0,
            step=0.01,
            description='',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        lambda_slider.observe(_on_select_lambda, names='value')
        lambda_slider_header = wdg.HTML('Lambda value:')
        lambda_slider_widget = wdg.HBox([lambda_slider_header, lambda_slider])
        words_plot = plot_terms(**_words_kws)
        words_plot_output = wdg.Output()
        words_plot_output.append_display_data(words_plot)
        words_plot_children = [words_header, lambda_slider_widget]\
            if show_headers else [lambda_slider_widget]
        words_plot_children.append(words_plot_output)
        words_widget = wdg.VBox(words_plot_children)
        children.append(words_widget)

    # Docs
    if show_docs:
        def _on_select_docs_num(sel):
            docs_num = docs_num_slider.value
            docs_plot_output.clear_output(wait=False)
            with docs_plot_output:
                _top_docs_kws.update({'docs_num': docs_num})
                top_docs = get_top_docs(**_top_docs_kws)
                _docs_kws.update({'docs': top_docs})
                display(plot_docs(**_docs_kws))

        docs_num_slider = wdg.IntSlider(
            value=2,
            min=1,
            max=100,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
        docs_num_slider.observe(_on_select_docs_num, names='value')
        docs_num_slider_header = wdg.HTML('Documents number:')
        docs_num_slider_widget = wdg.HBox(
            [docs_num_slider_header, docs_num_slider])

        docs_plot = plot_docs(**_docs_kws)
        docs_plot_output = wdg.Output()
        docs_plot_output.append_display_data(docs_plot)
        docs_plot_children = [docs_header, docs_num_slider_widget]\
            if show_headers else [docs_num_slider_widget]
        docs_plot_children.append(docs_plot_output)
        docs_widget = wdg.VBox(docs_plot_children)
        children.append(docs_widget)

    grid_box = wdg.GridBox(children, layout=layout)
    hr = wdg.HTML('<hr style="height:1px;border:none;color:#333;background-color:#333;" />')
    app = wdg.VBox(
        [select_topic_widget, hr, grid_box],
        layout={'align_items': 'center'})

    return app
