__all__ = ['prepare_coords', 'report']
from typing import Optional, Sequence, List
from copy import deepcopy
from ipywidgets import widgets as wdg
from pandas import DataFrame
from ._distance import get_topics_dist, get_topics_scatter
from ._vis import plot_scatter_topics, plot_terms, plot_docs
from ._helpers import (
    calc_terms_probs_ratio,
    get_phi, get_theta,
    get_top_docs)


def prepare_coords(
        model: object,
        labels: Optional[Sequence] = None,
        dist_kws: dict = None,
        scatter_kws: dict = None) -> DataFrame:
    """Prepare coordinates for topics scatter plot.

    Parameters
    ----------
    model : object
        Topic model instance.
    labels : Optional[Sequence]
        Topics labels.
    dist_kws : dict, optional
        Keyword arguments passed to :py:meth:`tmplot.get_topics_dist()`.
    scatter_kws : dict, optional
        Keyword arguments passed to :py:meth:`tmplot.get_topics_scatter()`.
    """
    if not dist_kws:
        dist_kws = {}
    if not scatter_kws:
        scatter_kws = {}

    phi = get_phi(model)
    theta = get_theta(model)
    topics_dists = get_topics_dist(phi, **dist_kws)
    topics_coords = get_topics_scatter(topics_dists, theta, **scatter_kws)
    topics_coords['label'] = labels or theta.index
    return topics_coords


def report(
        model: object = None,
        docs: Optional[Sequence[str]] = None,
        topics_labels: Optional[Sequence[str]] = None,
        corpus: Optional[List] = None,
        layout: wdg.Layout = None,
        show_headers: bool = True,
        show_docs: bool = True,
        show_words: bool = True,
        show_topics: bool = True,
        topics_kws: dict = None,
        height: int = 500,
        width: int = 250,
        coords_kws: dict = None,
        words_kws: dict = None,
        docs_kws: dict = None,
        top_docs_kws: dict = None) -> wdg.VBox:
    """Interactive report interface.

    Parameters
    ----------
    model : object, optional
        Topic model instance.
    docs : Optional[Sequence[str]], optional
        Documents.
    topics_labels : Optional[Sequence[str]], optional
        Topics labels.
    corpus : Optional[List[str]], optional
        Gensim corpus.
    layout : wdg.Layout, optional
        Interface layout instance.
    show_headers : bool, optional
        Show headers.
    show_docs : bool, optional
        Show documents widget.
    show_words : bool, optional
        Show words widget.
    show_topics : bool, optional
        Show topics scatter plot widget.
    topics_kws : dict, optional
        Keyword arguments passed to :py:meth:`tmplot.plot_scatter_topics()`.
    coords_kws : dict, optional
        Keyword arguments passed to :py:meth:`tmplot.prepare_coords()`.
    words_kws : dict, optional
        Keyword arguments passed to :py:meth:`tmplot.plot_terms()`.
    docs_kws : dict, optional
        Keyword arguments passed to :py:meth:`tmplot.plot_docs()`.
    top_docs_kws : dict, optional
        Keyword arguments passed to :py:meth:`tmplot.get_top_docs()`.

    Returns
    -------
    wdg.VBox
        Report interface as a VBox instance.
    """
    from IPython.display import display

    _topics_kws = {
        'chart_kws': {'height': height, 'width': width}}\
        if not topics_kws else deepcopy(topics_kws)
    _coords_kws = {} if not coords_kws else deepcopy(coords_kws)
    _words_kws = {
        'chart_kws': {'height': height, 'width': width}}\
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
            'size_col': 'size',
            'topic': 0
        })

    if 'terms_probs' not in _words_kws:
        phi = get_phi(model)
        terms_probs = calc_terms_probs_ratio(phi, topic=0)
        _words_kws.update({'terms_probs': terms_probs})

    if 'docs' not in _docs_kws:
        theta = get_theta(model, corpus=corpus).values
        _top_docs_kws.update({
            'docs': docs, 'theta': theta,
            'topics': [0], 'docs_num': 2})
        top_docs = get_top_docs(**_top_docs_kws)
        top_docs.columns = ['']
        _docs_kws.update({'docs': top_docs})

    # Topic selection
    def _on_select_topic(sel):
        topic = sel['new']
        topics_plot_output.clear_output(wait=False)
        words_plot_output.clear_output(wait=False)
        docs_plot_output.clear_output(wait=False)
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
            top_docs.columns = ['']
            _docs_kws.update({'docs': top_docs})
            display(plot_docs(**_docs_kws))

    topics_ids = list(range(len(_topics_kws['topics_coords'])))
    topics_labels = topics_labels or topics_ids
    select_topic = wdg.Dropdown(
        options=list(zip(topics_labels, topics_ids)), value=0)
    select_topic.observe(_on_select_topic, names='value')
    select_topic_header = wdg.HTML('<b>Select a topic</b>:')
    select_topic_widget = wdg.HBox([select_topic_header, select_topic])
    select_topic_wrapper = wdg.VBox(
        [select_topic_widget], layout={'align_items': 'center'})

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
            ('TSNE', 'tsne'),
            ('SpectralEmbedding', 'sem'),
            ('MDS', 'mds'),
            ('LocallyLinearEmbedding', 'lle'),
            ('Isomap', 'isomap')
        ]
        topics_method_header = wdg.HTML('Select a method:')
        topics_method = wdg.Dropdown(
            options=options_methods,
            value='tsne',
            layout=wdg.Layout(width=f'{width/1.25}px')
        )
        topics_method_widget = wdg.HBox([topics_method_header, topics_method])
        topics_method.observe(_on_select_topics_method, names='value')
        topics_plot_output = wdg.Output()
        topics_plot = plot_scatter_topics(**_topics_kws)
        topics_plot_output.append_display_data(topics_plot)
        topics_plot_children.extend([topics_method_widget, topics_plot_output])
        topics_widget = wdg.VBox(
            topics_plot_children,
            layout={'align_items': 'center'})
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
            value=0.6,
            min=0.0,
            max=1.0,
            step=0.01,
            description='',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=wdg.Layout(width=f'{width/1.25}px')
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
        words_widget = wdg.VBox(
            words_plot_children,
            layout={'align_items': 'center'})
        children.append(words_widget)

    # Docs
    if show_docs:
        def _on_select_docs_num(_):
            docs_num = docs_num_slider.value
            docs_plot_output.clear_output(wait=False)
            with docs_plot_output:
                _top_docs_kws.update({'docs_num': docs_num})
                top_docs = get_top_docs(**_top_docs_kws)
                top_docs.columns = ['']
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
            layout=wdg.Layout(width=f'{width/1.25}px')
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
        docs_widget = wdg.VBox(
            docs_plot_children,
            layout={'align_items': 'center'})
        children.append(docs_widget)

    grid_box = wdg.GridBox(children, layout=layout)
    hr_line = wdg.HTML('<hr style="border: 0; border-bottom: 1px solid #aaa">')
    app = wdg.VBox([select_topic_wrapper, hr_line, grid_box])

    return app
