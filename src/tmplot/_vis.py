# TODO: interactive topic model report: scatter plot of topics, top words/docs in topic (number is controlled interactively)
# TODO: heatmap of docs in topics
# TODO: topic dynamics in time
# TODO: word cloud
from altair import Chart, X, Y, Size, Color, Tooltip, value, Text, Scale, Legend
from numpy import ndarray
from pandas import DataFrame
from typing import Union


def plot_scatter_topics(
        topics_coords: Union[ndarray, DataFrame],
        x_col: str = "x",
        y_col: str = "y",
        size_col: str = None,
        label_col: str = None,
        color_col: str = None,
        topic_col: str = None,
        x_kws: dict = {},
        y_kws: dict = {},
        circle_kws: dict = {},
        circle_enc_kws: dict = {},
        text_kws: dict = {},
        text_enc_kws: dict = {},
        size_kws: dict = {},
        color_kws: dict = {},
        ) -> Chart:
    data = DataFrame(topics_coords, columns=[x_col, y_col])\
        if isinstance(topics_coords, ndarray)\
        else topics_coords.copy()

    if not topic_col:
        topic_col = "topic"
        data = data.assign(**{topic_col: range(1, len(topics_coords) + 1)})

    circle_kws.setdefault("opacity", 0.33)
    circle_kws.setdefault("stroke", 'black')
    circle_kws.setdefault("strokeWidth", 1)

    x_kws.update({'shorthand': x_col})
    x_kws.setdefault('axis', None)

    y_kws.update({'shorthand': y_col})
    y_kws.setdefault('axis', None)

    size_kws.setdefault('title', 'Marginal topic distribution')
    size_kws.setdefault('scale', Scale(range=[0, 3000]))
    circle_enc_kws.setdefault("x", X(**x_kws))
    circle_enc_kws.setdefault("y", Y(**y_kws))
    circle_enc_kws.setdefault(
        "size",
        Size(size_col, **size_kws) if size_col else value(500))

    text_kws.setdefault("align", "center"),
    text_kws.setdefault("baseline", "middle")

    text_enc_kws.setdefault("x", X(**x_kws))
    text_enc_kws.setdefault("y", Y(**y_kws))
    text_enc_kws.setdefault("text", Text(topic_col))
    text_enc_kws.setdefault("size", value(12))

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
        .configure_legend(orient='bottom')


def plot_terms(
        terms_probs: DataFrame,
        x_col: str = 'Probability',
        y_col: str = 'Terms',
        color_col: str = 'Type',
        chart_kws: dict = {},
        bar_kws: dict = {},
        x_kws: dict = {},
        y_kws: dict = {},
        color_kws: dict = {}) -> Chart:
    x_kws.setdefault('stack', None)
    y_kws.setdefault('title', None)
    bar_kws.setdefault('opacity', 0.5)
    color_kws.setdefault('legend', Legend(orient='bottom'))

    return Chart(terms_probs, **chart_kws).mark_bar(**bar_kws).encode(
        x=X(x_col, **x_kws),
        y=Y(y_col, **y_kws),
        color=Color(color_col, **color_kws)
    )
