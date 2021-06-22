# TODO: interactive topic model report: scatter plot of topics, top words/docs in topic (number is controlled interactively)
# TODO: heatmap of docs in topics
# TODO: topic dynamics in time
# TODO: word cloud
from typing import Union, Sequence
from pandas import DataFrame, option_context
from numpy import ndarray
from altair import (
    Chart, X, Y, Size, Color, Tooltip, value, Text, Scale, Legend)


def plot_scatter_topics(
        topics_coords: Union[ndarray, DataFrame],
        x_col: str = "x",
        y_col: str = "y",
        size_col: str = None,
        label_col: str = None,
        color_col: str = None,
        topic_col: str = None,
        font_size: int = 12,
        x_kws: dict = None,
        y_kws: dict = None,
        circle_kws: dict = None,
        circle_enc_kws: dict = None,
        text_kws: dict = None,
        text_enc_kws: dict = None,
        size_kws: dict = None,
        color_kws: dict = None) -> Chart:

    if not x_kws:
        x_kws = {'shorthand': x_col, 'axis': None}

    if not y_kws:
        y_kws = {'shorthand': y_col, 'axis': None}

    if not circle_kws:
        circle_kws = {"opacity": 0.33, "stroke": 'black', "strokeWidth": 1}

    if not size_kws:
        size_kws = {
            'title': 'Marginal topic distribution',
            'scale': Scale(range=[0, 3000])}

    if not circle_enc_kws:
        circle_enc_kws = {
            "x": X(**x_kws),
            "y": Y(**y_kws),
            "size": Size(size_col, **size_kws) if size_col else value(500)}

    if not text_kws:
        text_kws = {"align": "center", "baseline": "middle"}

    if not text_enc_kws:
        text_enc_kws = {
            "x": X(**x_kws),
            "y": Y(**y_kws),
            "text": Text(topic_col),
            "size": value(font_size)}

    if not color_kws:
        color_kws = {}

    data = DataFrame(topics_coords, columns=[x_col, y_col])\
        if isinstance(topics_coords, ndarray)\
        else topics_coords.copy()

    if not topic_col:
        topic_col = "topic"
        data = data.assign(**{topic_col: range(1, len(topics_coords) + 1)})

    if label_col:
        circle_enc_kws.update({'tooltip': Tooltip(label_col, **size_kws)})
        text_enc_kws.update({'tooltip': Tooltip(label_col, **size_kws)})
    if color_col:
        circle_enc_kws.update({'color': Color(color_col, **color_kws)})

    base = Chart(data)

    rule = base\
        .mark_rule()\
        .encode(
            y='average(y)',
            color=value('gray'),
            size=value(0.2))

    rule2 = base\
        .mark_rule()\
        .encode(
            x='average(x)',
            color=value('gray'),
            size=value(0.2))

    points = base\
        .mark_circle(**circle_kws)\
        .encode(**circle_enc_kws)

    text = base\
        .mark_text(**text_kws)\
        .encode(**text_enc_kws)

    return (rule + rule2 + points + text)\
        .configure_view(stroke='transparent')\
        .configure_legend(
            orient='bottom',
            labelFontSize=font_size,
            titleFontSize=font_size)\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)


def plot_terms(
        terms_probs: DataFrame,
        x_col: str = 'Probability',
        y_col: str = 'Terms',
        color_col: str = 'Type',
        font_size: int = 12,
        chart_kws: dict = {},
        bar_kws: dict = {},
        x_kws: dict = {},
        y_kws: dict = {},
        color_kws: dict = {}) -> Chart:
    x_kws.setdefault('stack', None)
    y_kws.setdefault('sort', None)
    y_kws.setdefault('title', None)
    # bar_kws.setdefault('opacity', 1)
    color_kws.setdefault('legend', Legend(orient='bottom'))
    color_kws.setdefault('scale', Scale(scheme='category20'))

    return Chart(terms_probs, **chart_kws)\
        .mark_bar(**bar_kws)\
        .encode(
            x=X(x_col, **x_kws),
            y=Y(y_col, **y_kws),
            color=Color(color_col, **color_kws)
        )\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)


def plot_docs(
        docs: Union[Sequence[str], DataFrame],
        styles: dict = None,
        html_kws: dict = None) -> DataFrame:
    from IPython.display import HTML

    if styles is None:
        styles = '<style>.plot{font-size: 1.1em !important;}' +\
            'table td{text-align: left !important}' +\
            'table th{text-align: center !important}</style>'
    if html_kws is None:
        html_kws = {'classes': 'plot'}

    if isinstance(docs, DataFrame):
        df_docs = docs.copy()
    else:
        df_docs = DataFrame({'docs': docs})

    with option_context('display.max_colwidth', 0):
        df_docs.style.set_properties(**{'text-align': 'center'})
        return HTML(styles + df_docs.to_html(**html_kws))
