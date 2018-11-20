"""Module that handles creation of incremental first order oracles for different domains including
    binary classification or structured prediction tasks such as named entity recognition and object localization.

.. moduleauthor:: Krishna Pillutla <last-name AT cs DOT washington DOT edu>

"""

from .classification import LogisticRegressionIfo
from . import named_entity_recognition, localization

__all__ = ['LogisticRegressionIfo', 'named_entity_recognition', 'localization']