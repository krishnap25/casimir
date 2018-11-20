"""Core module with optimization algorithms and abstract classes representing
    Incremental First Order Oracles.

.. moduleauthor:: Krishna Pillutla <last-name AT cs DOT washington DOT edu>

"""

from .incremental_first_order_oracle import IncrementalFirstOrderOracle, SmoothedIncrementalFirstOrderOracle
from .regularization import SmoothRegularizationPenalty, L2Penalty
from .optimization_algorithms import Optimizer, CasimirSVRGOptimizer, SGDOptimizer, SVRGOptimizer
from .optimization_algorithms import optimize_ifo, block_coordinate_frank_wolfe_optimize, termination_gradient_norm

__all__ = ['IncrementalFirstOrderOracle', 'SmoothedIncrementalFirstOrderOracle', 'SmoothRegularizationPenalty',
           'L2Penalty', 'optimize_ifo', 'Optimizer', 'CasimirSVRGOptimizer',
           'SGDOptimizer', 'SVRGOptimizer', 'block_coordinate_frank_wolfe_optimize', 'termination_gradient_norm']
