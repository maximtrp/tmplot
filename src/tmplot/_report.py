from typing import Iterable
from ipywidgets import widgets as wdg
from numpy import ndarray
from ._vis import plot_scatter_topics, plot_terms, plot_docs


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

    for active, header, func, kwargs in zip(show, headers, funcs, funcs_args):
        if active:
            plot = func(**funcs_args)
            plot_children = [header] if show_headers else []
            plot_children.append(plot)
            plot_widget = wdg.VBox(plot_children)
            children.append(plot_widget)

    grid_box = wdg.GridBox(children, layout=layout)

    return grid_box
