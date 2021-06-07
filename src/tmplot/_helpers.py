__all__ = ['get_phi']
# TODO: load phi and theta matrices from different models/packages
from numpy import ndarray
from pandas import concat, Series
from tomotopy import LDAModel, LLDAModel
from functools import partial


def get_phi(
        model: object) -> ndarray:
    """Returns topics (rows) vs words (cols) matrix"""

    tomotopy_models = [LDAModel, LLDAModel]
    _isinstance = partial(isinstance, model)

    if any(map(_isinstance, tomotopy_models)):
        _twd = map(lambda x: Series(model.get_topic_word_dist(x)), range(model.k))
        phi = concat(_twd, axis=1)
        phi.index = model.vocabs

    return phi