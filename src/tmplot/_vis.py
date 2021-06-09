# TODO: interactive topic model report: scatter plot of topics, top words/docs in topic (number is controlled interactively)
# TODO: heatmap of docs in topics
# TODO: topic dynamics in time
# TODO: word cloud
from altair import Chart, X, Y, Size, Color, Tooltip, value, Text
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

    circle_kws.setdefault("opacity", 0.5)
    circle_kws.setdefault("stroke", 'black')
    circle_kws.setdefault("strokeWidth", 1)

    x_kws.update({'shorthand': x_col})
    x_kws.setdefault('axis', None)

    y_kws.update({'shorthand': y_col})
    y_kws.setdefault('axis', None)

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

    points = base\
        .mark_circle(**circle_kws)\
        .encode(**circle_enc_kws)

    text = base\
        .mark_text(**text_kws)\
        .encode(**text_enc_kws)

    return (points + text).configure_view(stroke='transparent')
