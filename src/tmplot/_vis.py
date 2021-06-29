# TODO: interactive topic model report: scatter plot of topics,
#       top words/docs in topic (number is controlled interactively)
# TODO: heatmap of docs in topics
# TODO: topic dynamics in time
# TODO: word cloud
__all__ = [
    'plot_scatter_topics', 'plot_terms', 'plot_docs']
from typing import Union, Sequence
from pandas import DataFrame, option_context
from numpy import ndarray
from altair import (
    Chart, X, Y, Size, Color, value, Text, Scale, Legend)


def plot_scatter_topics(
        topics_coords: Union[ndarray, DataFrame],
        x_col: str = "x",
        y_col: str = "y",
        topic: int = None,
        size_col: str = None,
        label_col: str = None,
        color_col: str = None,
        topic_col: str = None,
        font_size: int = 13,
        x_kws: dict = None,
        y_kws: dict = None,
        chart_kws: dict = None,
        circle_kws: dict = None,
        circle_enc_kws: dict = None,
        text_kws: dict = None,
        text_enc_kws: dict = None,
        size_kws: dict = None,
        color_kws: dict = None) -> Chart:
    """Topics scatter plot in 2D.

    Parameters
    ----------
    topics_coords : Union[ndarray, DataFrame]
        Topics scatter coordinates.
    x_col : str, optional
        X column name.
    y_col : str, optional
        Y column name.
    topic : int, optional
        Topic index.
    size_col : str, optional
        Column with size values.
    label_col : str, optional
        Column with topic labels.
    color_col : str, optional
        Column with colors.
    topic_col : str, optional
        Column with topics texts.
    font_size : int, optional
        Font size.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()`.
    circle_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_circle()`.
    circle_enc_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.encode()`
        for circle elements.
    text_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_text()`.
    text_enc_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.encode()`
        for text elements.
    size_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Size()` for
        circle elements.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()` for
        circle elements.

    Returns
    -------
    Chart
        Topics scatter plot.
    """
    if not chart_kws:
        chart_kws = {}

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

    if not color_kws:
        color_kws = {}\
            if topic is None\
            else {'condition': {
                "test": f"datum['topic'] == {topic}", "value": "red"}}

    data = DataFrame(topics_coords, columns=[x_col, y_col])\
        if isinstance(topics_coords, ndarray)\
        else topics_coords.copy()

    if not topic_col:
        topic_col = "topic"
        data = data.assign(**{topic_col: range(len(topics_coords))})

    if not text_enc_kws:
        text_enc_kws = {
            "x": X(**x_kws),
            "y": Y(**y_kws),
            "text": Text(topic_col),
            "size": value(font_size)}

    # Tooltips initialization
    tooltips = []
    if label_col:
        tooltips.append(label_col)
    if size_col:
        tooltips.append(size_col)

    if tooltips:
        circle_enc_kws.update({'tooltip': tooltips})
        text_enc_kws.update({'tooltip': tooltips})

    if color_kws:
        circle_enc_kws.update({'color': Color(**color_kws)})

    base = Chart(data, **chart_kws)

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
        font_size: int = 13,
        chart_kws: dict = None,
        bar_kws: dict = None,
        x_kws: dict = None,
        y_kws: dict = None,
        color_kws: dict = None) -> Chart:
    """Plot words conditional and marginal probabilities.

    Parameters
    ----------
    terms_probs : DataFrame
        Words probabilities.
    x_col : str, optional
        X column name.
    y_col : str, optional
        Y column name.
    color_col : str, optional
        Column with values types (for coloring).
    font_size : int, optional
        Font size.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()`.
    bar_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_bar()`.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()`.

    Returns
    -------
    Chart
        Terms probabilities chart.
    """
    if not x_kws:
        x_kws = {'stack': None}
    if not y_kws:
        y_kws = {'sort': None, 'title': None}
    if not color_kws:
        color_kws = {
            'shorthand': color_col,
            'legend': Legend(orient='bottom'),
            'scale': Scale(scheme='category20')
        }
    if not chart_kws:
        chart_kws = {}
    if not bar_kws:
        bar_kws = {}

    return Chart(data=terms_probs, **chart_kws)\
        .mark_bar(**bar_kws)\
        .encode(
            x=X(x_col, **x_kws),
            y=Y(y_col, **y_kws),
            color=Color(**color_kws)
        )\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(
            labelFontSize=font_size, titleFontSize=font_size,
            columns=1, labelLimit=250)


def plot_docs(
        docs: Union[Sequence[str], DataFrame],
        styles: str = None,
        html_kws: dict = None) -> DataFrame:
    """Documents plotting functionality for report interface.

    Parameters
    ----------
    docs : Union[Sequence[str], DataFrame]
        Documents.
    styles : str, optional
        Styles string for formatting the table with documents.
        Concatenated with HTML.
    html_kws : dict, optional
        Keyword arguments passed to :py:meth:`pandas.DataFrame.to_html` method.

    Returns
    -------
    ipywidgets.HTML
        Topic documents.
    """
    from IPython.display import HTML

    if styles is None:
        styles = '<style>table td{text-align: left !important}' +\
            'table th{text-align: center !important}</style>'
    if html_kws is None:
        # html_kws = {'classes': 'plot'}
        html_kws = {}

    if isinstance(docs, DataFrame):
        df_docs = docs.copy()
    else:
        df_docs = DataFrame({'docs': docs})

    with option_context('display.max_colwidth', 0):
        # df_docs.style.set_properties(**{'text-align': 'center'})
        return HTML(styles + df_docs.to_html(**html_kws))
