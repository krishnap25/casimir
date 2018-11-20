"""Read pre-processed data of Pascal VOC 2007 and create smooth incremental first order oracles for Localization.

.. moduleauthor:: Krishna Pillutla <last-name AT cs DOT washington DOT edu>

"""

from .incremental_first_order_oracle import LocalizationIfo, create_loc_ifo_from_data
from .reader import VocDataset

__all__ = ['create_loc_ifo_from_data', 'LocalizationIfo', 'VocDataset']
