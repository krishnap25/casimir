API reference for casimir.optim
==============================================

Incremental First Order Oracles
-------------------------------
.. autoclass:: casimir.optim.IncrementalFirstOrderOracle
    :members:
.. autoclass:: casimir.optim.SmoothedIncrementalFirstOrderOracle
    :members:

Optimization Algorithms
-----------------------
.. autofunction:: casimir.optim.optimize_ifo
.. autoclass:: casimir.optim.Optimizer
    :members:
.. autoclass:: casimir.optim.CasimirSVRGOptimizer
    :members:
.. autoclass:: casimir.optim.SGDOptimizer
    :members:
.. autoclass:: casimir.optim.SVRGOptimizer
    :members:
.. autofunction:: casimir.optim.block_coordinate_frank_wolfe_optimize
.. autofunction:: casimir.optim.termination_gradient_norm

Regularization
--------------
.. autoclass:: casimir.optim.L2Penalty
    :members:
