"""
.. module:: optimization_algorithms
   :synopsis: Implementation of optimization algorithms with access to (smooth) incremental first order oracles.


.. moduleauthor:: Krishna Pillutla <last-name AT cs DOT washington DOT edu>

"""
from __future__ import absolute_import, division, print_function
import copy
import math
import numpy as np
import sys
import time
from . import L2Penalty


def optimize_ifo(initial_model, train_ifo, dev_ifo=None, test_ifo=None, algorithm='SGD',
                 reg_penalties=None, num_passes=100, termination_criterion=None, seed=25,
                 logging=True, verbose=True, optim_options=None):
    """Minimize a convex function with access to a (smoothed) incremental first order oracle using a primal algorithm.

    Functions that can be handled are of the form :math:`f(w) = \\frac{1}{n} \\sum_{i=1}^n f_i(w) + r(w)`,
    where a (smoothed) incremental first order oracle gives access to each :math:`f_i` and :math:`r(w)` is a (sum of)
    regularization penalties. The argument ``algorithm`` controls which optimization algorithm is used.

    :param ``numpy.ndarray`` initial_model: Starting iterate of the optimization algorithm.
    :param train_ifo: (Smoothed) Incremental First Order Oracle for the function to minimize.
    :type train_ifo: :class:`incremental_first_order_oracle.IncrementalFirstOrderOracle` or
        :class:`incremental_first_order_oracle.SmoothIncrementalFirstOrderOracle`
    :param dev_ifo: Incremental First Order Oracle for development set.
        A value of ``None`` indicates no development set.
    :type dev_ifo: :class:`incremental_first_order_oracle.IncrementalFirstOrderOracle` or
        :class:`incremental_first_order_oracle.SmoothIncrementalFirstOrderOracle` or ``None``
    :param test_ifo: Incremental First Order Oracle for test set. A value of ``None`` indicates no test set.
    :type test_ifo: :class:`incremental_first_order_oracle.IncrementalFirstOrderOracle` or
        :class:`incremental_first_order_oracle.SmoothIncrementalFirstOrderOracle` or ``None``
    :param algorithm: Optimization algorithm to use. Should be from the list ``['sgd', 'svrg', 'casimir_svrg']``.
    :param seed: Integer to be used as random seed. Default 25.
    :param reg_penalties: List of regularization penalties where
        :math:`r(w)` corresponds to sum of penalties in the list. Default ``None``, in which case no regularization
        penalty is applied.
    :type reg_penalties: List of :class:`regularization.SmoothRegularizationPenalty` or ``None``
    :param num_passes: Maximum allowed number of passes through the data. Default 100.
    :param termination_criterion: ``None`` or Callable, takes two arguments `model`, `train_ifo`
        and returns ``True`` if converged.
        If ``None``, the given iteration budget ``num_passes`` is used to terminate optimization. Default ``None``.
        See :py:func:`termination_gradient_norm` for an example.
    :param logging: (boolean) Log performance of optimization algorithm. Default ``True``.
    :param verbose: (boolean) Print performance of optimization algorithm, only if ``logging`` is ``True``.
        Default ``True``.
    :param dict optim_options: Algorithm-specific options. Default ``None``.
        Consult named parameters of specific optimizers for allowed options.
    :return: The tuple (final iterate, logs) where logs are a list of lists, one for each epoch. The list contains
        the epoch number, train function value, dev function value, dev evaluation metric, test function value, test
        evaluation metric and the time taken to perform this epoch. The dev (test) statistics are printed only if
        ``dev_ifo`` (resp. ``test_ifo``) is not ``None``.
    """

    model = initial_model
    rng = np.random.RandomState(seed)

    # default options
    if reg_penalties is None:
        reg_penalties = []
    if optim_options is None:
        optim_options = {}

    # create optimizer
    optimizer = _get_optimizer(algorithm)(initial_model, reg_penalties, **optim_options)

    # logging
    start_time = time.time()
    log_output = []

    if logging:
        _log(log_output, 0, model, reg_penalties, train_ifo, dev_ifo, test_ifo, verbose, start_time)

    # main optimization loop
    for epoch in range(num_passes):
        start_time = time.time()

        # check termination
        if termination_criterion is not None and termination_criterion(model, train_ifo):
            if verbose:
                print('Exiting after {} epochs because termination criterion has been met.'.format(epoch))
            break

        # start epoch
        optimizer.start_epoch(train_ifo, epoch)

        # loop over tests
        for i in range(len(train_ifo)):
            iteration = epoch * len(train_ifo) + i  # inner iteration number
            # idx = i  # for testing
            idx = rng.randint(0, len(train_ifo))  # index of random example
            optimizer.step(train_ifo, idx, iteration)

        # log performance of current model
        model = optimizer.end_epoch()
        if logging:
            _log(log_output, epoch+1, model, reg_penalties, train_ifo, dev_ifo, test_ifo, verbose, start_time)

    return model, log_output


# Termination criteria
def termination_gradient_norm(model, train_ifo, gradient_norm_tolerance=1e-5):
    """Terminate optimization after gradient norm falls below a certain tolerance.

    :param model: Current iterate of optimizer
    :param train_ifo: (Smoothed) Incremental first order oracle.
    :param gradient_norm_tolerance: Tolerance.
    :return: True if gradient norm is smaller than tolerance, false otherwise.
    """
    grad = train_ifo.batch_gradient(model)
    grad_norm = np.linalg.norm(grad.reshape(-1))
    return grad_norm < gradient_norm_tolerance


# Optimization algorithms

class Optimizer(object):
    """Base class for optimizers using (smoothed) incremental first order oracles.

        Optimizer classes store the state of the variables to be optimized.

        Any subclass must override the methods ``start_epoch``, ``step``, ``end_epoch``.

    """
    def __init__(self, initial_model, reg_penalties=None):
        """

        :param initial_model: Initial iterate of optimizer.
        :param reg_penalties: List of smooth regularization penalties.
        """
        self.model = np.asarray(copy.deepcopy(initial_model))
        self.reg_penalties = copy.deepcopy(reg_penalties) if reg_penalties is not None else []

    def start_epoch(self, train_ifo, epoch):
        """Prepare for start of epoch.

        Example uses are warm start in Casimir-SVRG or batch gradient computation in SVRG. Adaptive smoothing algorithms
        may update smoothing parameter here.

        :param: train_ifo: (Smoothed) Incremental First Order Oracle for the training set.
            This object may be mutated by adaptive smoothing algorithms.
        """
        raise NotImplementedError

    def step(self, train_ifo, idx, iteration):
        """Take a step based on sample indexed by ``idx``.

        :param train_ifo: (Smoothed) Incremental First Order Oracle for the training set.
        :param idx: Index of sample to use to take a step.
        :param iteration: Global iteration number. Required, e.g., by :class:`SGDOptimizer` to compute learning rate.
        """
        raise NotImplementedError

    def end_epoch(self):
        """Perform optional computation to end epoch and return current iterate to be used for logging.

        :return: model, denoting optimizer state at current time.
        """
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    """Implement the stochastic (sub-)gradient method with various learning rate and averaging schemes.


    This optimizer can be invoked by calling :py:func:`optimize_ifo` with argument
    ``algorithm='sgd'``.

    :param initial_model: Initial iterate of optimizer.
    :param reg_penalties: List of smooth regularization penalties.

    The following named parameters can be passed though ``optim_options`` of :py:func:`optimize_ifo`:

    :param learning_rate_scheme: Learning rate schemes. Allowed values are ``'const'``, ``'pegasos'``
        or ``'linear'``. See below for a description.
    :param averaging: Use parameter averaging. Allowed values are ``'none'``, ``'uavg'``, ``'wavg'``.
        See below for a description.
    :param initial_learning_rate: The parameter :math:`\eta_0` used to determine the learning rate.
    :param learning_rate_time_factor: The parameter :math:`t_0` used to determine the learning rate.

    Allowed values of the parameter ``learning_rate_scheme``
    and corresponding learning rates :math:`\eta_t` at iteration t are:

    - ``'const'`` (default): :math:`\eta_t = \eta_0`,
    - ``'pegasos'``: :math:`\eta_t = 1/(\lambda t)`,
    - ``'linear'``: :math:`\eta_t = \eta_0 / (1 + t/t_0)`

    Here, :math:`\eta_0, t_0` are parameters described above and :math:`\lambda` the strong convexity.

    Allowed values of the parameter ``averaging`` and the corresponding averaged iterates used are:

    - ``'none'``: no averaging, use final iterate
    - ``'uavg'``: uniform average :math:`\\bar w_T = \\frac{1}{T+1}\\sum_{t=0}^{T} w_t`
    - ``'wavg'`` (default): weighted average :math:`\\bar w_T = \\frac{2}{T(T+1)} \\sum_{t=0}^{T} t\, w_t`
    """
    def __init__(self, initial_model, reg_penalties=None,
                 learning_rate_scheme='const', averaging='wavg',
                 initial_learning_rate=1.0, learning_rate_time_factor=100.0):
        super(SGDOptimizer, self).__init__(initial_model, reg_penalties)
        self.strong_convexity = _get_strong_convexity_from_regularization(reg_penalties)
        self.learning_rate_function = _get_learning_rate_function_sgd(learning_rate_scheme, self.strong_convexity,
                                                                      initial_learning_rate, learning_rate_time_factor)

        self.u_avg_model = copy.deepcopy(self.model)  # uniform average
        self.w_avg_model = copy.deepcopy(self.model)  # weighted average
        if averaging == 'none':
            self.avg_model = self.model
        elif averaging == 'uavg':
            self.avg_model = self.u_avg_model
        elif averaging == 'wavg':
            self.avg_model = self.w_avg_model
        else:
            raise ValueError('Unknown averaging for SGD,', averaging)
        self.count = 0.0

    def start_epoch(self, train_ifo, epoch):
        """Do nothing."""
        pass

    def step(self, train_ifo, idx, iteration):
        """Perform a single SGD step and update averaged iterates."""
        gradient = train_ifo.gradient(self.model, idx) + _get_regularization_gradient(self.reg_penalties, self.model)
        learning_rate = self.learning_rate_function(iteration)
        self.model -= learning_rate * gradient

        # update averages online (done in-place so as to not lose reference to self.avg_model)
        self.u_avg_model *= self.count / (self.count + 1)
        self.u_avg_model += self.model / (self.count + 1)
        self.w_avg_model *= self.count / (self.count + 2)
        self.w_avg_model += self.model * (2 / (self.count + 2))
        self.count += 1

    def end_epoch(self):
        """Return averaged iterate."""
        return self.avg_model


class SVRGOptimizer(Optimizer):
    """Implement Stochastic Variance Reduced Gradient (SVRG) with optional smoothing.

        This optimizer can be invoked by calling :py:func:`optimize_ifo` with argument
        ``algorithm='svrg'``.
        The following named parameters can be passed though ``optim_options`` of :py:func:`optimize_ifo`:
        ``learning_rate`` and ``smoothing_coefficient``.

    """
    def __init__(self, initial_model, reg_penalties, learning_rate=1.0, smoothing_coefficient=None):
        super(SVRGOptimizer, self).__init__(initial_model, reg_penalties)
        self.learning_rate = learning_rate
        self.smoothing_coefficient = smoothing_coefficient
        self.avg_model = copy.deepcopy(initial_model)
        self.support_model = copy.deepcopy(initial_model)
        self.batch_gradient = None
        self.count = 0.0

    def start_epoch(self, train_ifo, epoch):
        """Update smoothing coefficient, reset averaged iterate and update batch gradient."""
        # update smoothing coefficient in train_ifo (mutates train_ifo)
        if hasattr(train_ifo, 'update_smoothing_coefficient'):
            train_ifo.update_smoothing_coefficient(self.smoothing_coefficient)
        # update iterate and batch gradient
        self.avg_model.fill(0.0)
        np.copyto(self.support_model, self.model)
        self.batch_gradient = train_ifo.batch_gradient(self.support_model)
        self.count = 0.0

    def step(self, train_ifo, idx, iteration):
        """Take a single SVRG step and update averaged iterate."""
        gradient_current = (train_ifo.gradient(self.model, idx) +
                            _get_regularization_gradient(self.reg_penalties, self.model))
        gradient_support = train_ifo.gradient(self.support_model, idx)
        self.model -= self.learning_rate * (gradient_current - gradient_support + self.batch_gradient)

        # update average
        self.avg_model *= self.count / (self.count + 1)
        self.avg_model += self.model / (self.count + 1)
        self.count += 1

    def end_epoch(self):
        """Copy averaged iterate to main iterate and return averaged iterate."""
        np.copyto(self.model, self.avg_model)
        return self.model


class CasimirSVRGOptimizer(Optimizer):
    """Implement Casimir (Catalyst with smoothing) or Catalyst outer loop with SVRG as the inner loop.

        This optimizer can be invoked by calling :py:func:`optimize_ifo` with argument
        ``algorithm='casimir_svrg'`` or ``algorithm='catalyst_svrg'``.

        Use of smoothing:
            This class can mimic the original Catalyst algorithm, if
            ``initial_smoothing_coefficient`` and ``initial_moreau_coefficient`` are both set to ``None``,
            and the ``moreau_coefficient_update_rule`` is set to ``'const'``.
        Inexactness criterion:
            Each inner SVRG loop is run for one epoch over the data.


        :param initial_model: Initial iterate of optimizer.
        :param reg_penalties: List of smooth regularization penalties.

        The following named parameters can be passed though ``optim_options`` of :py:func:`optimize_ifo`:

        :param learning_rate: Learning rate to use for inner SVRG iterations.
            Either learning rate or smoothness parameter must be specified.
        :param grad_lipschitz_parameter: Estimate of Lipschitz constant of gradient.
            Inner SVRG iterations then use a learning rate of :math:`1/(L+\lambda + \kappa)`, where
            :math:`L` is the grad_lipschitz_parameter, :math:`\lambda` is the strong convexity of the regularization
            and :math:`\kappa` is the Moreau coefficient added by Casimir.
            Either learning rate or grad_lipschitz_parameter must be specified.
        :param initial_smoothing_coefficient: The initial amount of smoothing :math:`\mu_0`
            to add to non-smooth objectives. If ``None``, no smoothing is added.
        :param smoothing_coefficient_update_rule: The update rule that determines the amount of smoothing :math:`\mu_t`
            to add in epoch :math:`t`. Allowed inputs are ``'const'`` or ``'linear'`` or ``'expo'``.
            See below for an explanation.
        :param initial_moreau_coefficient: The initial weight :math:`\kappa_0` of the proximal term
            :math:`\\frac{\kappa_t}{2} \|w - z_{t-1}\|^2` added by Casimir, where :math:`z_t` is the prox-center.
            Default ``None``, in which case the value suggested by theory is used, provided :math:`L` has been specified,
            and 'const' is used as ``moreau_coefficient_update_rule``.
        :param moreau_coefficient_update_rule: The update rule that determines the weight :math:`\kappa_t` added by
            Casimir in epoch :math:`t`. The allowed values and the corresponding update rules are
            ``'const'``, which uses :math:`\kappa_t = \kappa_0` (default) and
            ``'linear'``, which uses :math:`\kappa_t = t \kappa_0`.
        :param warm_start: Warm start strategy to use to find the starting iterate of the next epoch.
            The allowed values are ``'prox-center'``, ``'prev-iterate'``, ``'extrapolation'``.
            See below for a description.

        Allowed values for ``smoothing_coefficient_update_rule``
                ``'const'``
                    :math:`\mu_t = \mu_0` (default)
                ``'linear'``
                    :math:`\mu_t = \mu_0 / t`
                ``'expo'``
                    :math:`\mu_t = \mu_0 c_t^t`, where
                    :math:`c_t = \\sqrt{1 - \\frac{1}{2}\\sqrt{\\frac{\lambda}{\lambda + \kappa_t}}}`. Here,
                    :math:`\lambda` is the strong convexity of the regularization
                    and :math:`\kappa_t` is the Moreau coefficient, the weight of the proximal term
                    added by Casimir in epoch :math:`t`.
        Allowed values for ``warm_start``
                ``'prox-center'``
                    Use :math:`z_{t-1}`, prox center of the proximal term
                    :math:`\\frac{\kappa_t}{2} \|w - z_{t-1}\|^2` added by Casimir (default)
                ``'prev-iterate'``
                    Use :math:`w_{t-1}`, the previous iterate
                ``'extrapolation```
                    Use :math:`w_{t-1} + \\frac{\kappa_t}{\kappa_t + \lambda}(z_{t-1} - z_{t-2})`

    """
    def __init__(self, initial_model, reg_penalties, learning_rate=None, grad_lipschitz_parameter=None,
                 initial_smoothing_coefficient=None, smoothing_coefficient_update_rule='const',
                 initial_moreau_coefficient=None, moreau_coefficient_update_rule='const',
                 warm_start='prox-center'):
        super(CasimirSVRGOptimizer, self).__init__(initial_model, reg_penalties)
        if learning_rate is None and grad_lipschitz_parameter is None:
            raise ValueError('{} requires specifying either learning_rate or smoothness_parameter'.format(
                self.__class__.__name__
            ))
        self.learning_rate = learning_rate
        self.grad_lipschitz_parameter = grad_lipschitz_parameter
        self.strong_convexity = _get_strong_convexity_from_regularization(reg_penalties)

        # smoothing coefficient
        self.initial_smoothing_coefficient = initial_smoothing_coefficient
        self.smoothing_update_function = _get_smoothing_update_function_casimir(smoothing_coefficient_update_rule,
                                                                                initial_smoothing_coefficient,
                                                                                self.strong_convexity)

        # Moreau coefficient
        self.initial_moreau_coefficient = initial_moreau_coefficient
        self.moreau_coefficient = None
        self.moreau_coefficient_next = None
        self.moreau_coefficient_update_rule = moreau_coefficient_update_rule
        self.moreau_coefficient_update_function = None

        # warm start
        if warm_start not in ['prox-center', 'prev-iterate', 'extrapolation']:
            raise ValueError('Unknown warm start {} in class {}'.format(warm_start, self.__class__.__name__))
        self.warm_start = warm_start

        # models
        self.avg_model = copy.deepcopy(initial_model)
        self.avg_model_prev = copy.deepcopy(self.avg_model)
        self.prox_center = copy.deepcopy(initial_model)
        self.prox_center_prev = copy.deepcopy(self.prox_center)
        self.support_model = copy.deepcopy(self.model)
        self.batch_gradient = None
        self.count = 0.0

        # other Casimir parameters
        self.alpha = None  # will be initialized later

    def _update_moreau_coefficient(self, n):
        if self.initial_moreau_coefficient is None:
            assert self.grad_lipschitz_parameter is not None, \
                '{} requires specifying either initial_moreau_coefficient or grad_lipschitz_parameter'.format(
                    self.__class__.__name__
                )
            temp = self.grad_lipschitz_parameter / (n+1)

            # use optimal value predicted by theory and set update rule to 'const'
            if temp > self.strong_convexity:
                self.initial_moreau_coefficient = temp - self.strong_convexity
            else:
                self.initial_moreau_coefficient = temp
            self.moreau_coefficient_update_rule = 'const'

        # initialize alpha, if not done yet
        if self.alpha is None:
            self.alpha = ((math.sqrt(5)-1)/2 if self.strong_convexity == 0.0 else
                          math.sqrt(self.strong_convexity / (self.strong_convexity + self.initial_moreau_coefficient)))

        if self.moreau_coefficient_update_function is None:
            self.moreau_coefficient_update_function = _get_moreau_coefficient_update_function_casimir(
                self.moreau_coefficient_update_rule, self.initial_moreau_coefficient
            )

    def start_epoch(self, train_ifo, epoch):
        """Update Moreau coefficient, smoothing coefficient, learning rate, warm start and batch gradient."""
        self._update_moreau_coefficient(len(train_ifo))

        # update Moreau coefficient
        self.moreau_coefficient = self.moreau_coefficient_update_function(epoch)
        self.moreau_coefficient_next = self.moreau_coefficient_update_function(epoch+1)

        # add Moreau envelope (prox term penalty)
        self.reg_penalties.append(L2Penalty(self.moreau_coefficient, self.prox_center))

        # update smoothing coefficient in train_ifo (mutates train_ifo)
        if hasattr(train_ifo, 'update_smoothing_coefficient'):
            train_ifo.update_smoothing_coefficient(self.smoothing_update_function(epoch, self.moreau_coefficient))

        # set learning rate, if necessary
        if self.grad_lipschitz_parameter is not None:
            self.learning_rate = 1.0 / (self.grad_lipschitz_parameter + self.strong_convexity + self.moreau_coefficient)

        # update model based on warm start strategy
        if self.warm_start == 'prox-center':
            np.copyto(self.model, self.prox_center)
        elif self.warm_start == 'extrapolation':
            factor = self.moreau_coefficient / (self.strong_convexity + self.moreau_coefficient)
            np.copyto(self.model, self.avg_model + factor * (self.prox_center - self.prox_center_prev))
        elif self.warm_start == 'prev-iterate':
            np.copyto(self.model, self.avg_model)

        # update support model and batch gradient
        np.copyto(self.support_model, self.model)
        self.batch_gradient = train_ifo.batch_gradient(self.support_model)
        self.avg_model.fill(0.0)
        self.count = 0.0

    def step(self, train_ifo, idx, iteration):
        """Take a single inner SVRG step and update averaged iterate."""
        gradient_current = (train_ifo.gradient(self.model, idx) +
                            _get_regularization_gradient(self.reg_penalties, self.model))
        gradient_support = train_ifo.gradient(self.support_model, idx)
        self.model -= self.learning_rate * (gradient_current - gradient_support + self.batch_gradient)

        # update average
        self.avg_model *= self.count / (self.count + 1)
        self.avg_model += self.model / (self.count + 1)
        self.count += 1

    def end_epoch(self):
        """Remove prox penalty, compute :math:`\alpha, \beta` and update averaged iterate."""
        # remove prox penalty to previous prox center
        self.reg_penalties.pop()

        # compute new alpha and beta
        self.alpha, beta = self._get_next_alpha_beta()

        np.copyto(self.prox_center_prev, self.prox_center)
        np.copyto(self.prox_center, self.avg_model)
        self.prox_center *= (1 + beta)
        self.prox_center -= beta * self.avg_model_prev
        np.copyto(self.avg_model_prev, self.avg_model)
        return self.avg_model

    def _get_next_alpha_beta(self):
        # shorthand for convenience
        a = self.alpha  # \alpha_{k-1}
        sc = self.strong_convexity  # \lambda
        r = self.moreau_coefficient + self.strong_convexity  # \kappa_{k} + \lambda
        r1 = self.moreau_coefficient_next + self.strong_convexity  # \kappa_{k+1} + \lambda

        a1 = (-a**2 * r + math.sqrt(a**4 * r**2 + 4 * r1 * (a**2 * r + a * sc))) / (2 * r1)  # \alpha_k
        b = a * (1 - a) * r / (a**2 * r + a1 * r1)  # \beta_k
        return a1, b


# BCFW
def block_coordinate_frank_wolfe_optimize(initial_model, train_ifo, dev_ifo=None, test_ifo=None,
                                          reg_penalties=None, num_passes=100, termination_criterion=None, seed=25,
                                          logging=True, verbose=True):
    """
    Implement the Block Coordinate Frank-Wolfe (BCFW) algorithm for structured prediction.

    This algorithm is not in the incremental first order oracle model. It requires the task loss in addition to
    first order information of the objective function. From an implementation point of view, it requires an IFO with the
    ``linear_minimization_oracle`` method implemented.

    Implemented here is Algorithm 4 of Lacoste-Julien et. al. Block-coordinate Frank-Wolfe optimization
    for structural SVMs (2012).

    :param initial_model: Starting point of the optimization algorithm.
    :param train_ifo: (Smoothed) Incremental First Order Oracle for the function to minimize.
    :param dev_ifo: Incremental First Order Oracle for development set. A value of ``None`` indicates no development set.
    :param test_ifo: Incremental First Order Oracle for test set. A value of ``None`` indicates no test set.
    :param seed: Integer to be used as random seed. Default 25.
    :param reg_penalties: List of regularization.SmoothRegularizationPenalty objects.
        Here, :math:`r(w)` corresponds to sum of penalties in the list.
        BCFW requires non-zero regularization, so ``reg_penalties`` must not be ``None``.
    :param num_passes: Maximum allowed number of passes through the data. Default 100.
    :param termination_criterion: ``None`` or ``Callable``, takes two arguments ``model``, ``train_ifo``
        and returns ``True`` if converged.
        If ``None``, the given iteration budget ``num_passes`` is used to terminate optimization. Default ``None``.
    :param logging: (boolean) Log performance of optimization algorithm. Default ``True``.
    :param verbose: (boolean) Print performance of optimization algorithm, only if ``logging`` is ``True``.
        Default ``True``.
    :return: The tuple (final iterate, logs).
    """

    if not hasattr(train_ifo, 'linear_minimization_oracle'):
        raise AttributeError('block_coordinate_frank_wolfe requires the train_ifo to implement' +
                             ' the linear_minimization_oracle method.')

    # TODO: use sparse vectors
    strong_convexity = _get_strong_convexity_from_regularization(reg_penalties)
    assert strong_convexity > 0, 'BCFW requires non-zero regularization'
    rng = np.random.RandomState(seed)
    model = np.zeros_like(initial_model)  # start from 0
    model_avg = copy.deepcopy(model)  # averaged iterate

    n = len(train_ifo)
    model_pts = np.empty((n,), dtype=np.object)
    losses = np.zeros((n,))
    for i in range(n):
        model_pts[i] = copy.deepcopy(model)
    count = 0.0

    # logging
    start_time = time.time()
    log_output = []
    if logging:
        _log(log_output, 0, model, reg_penalties, train_ifo, dev_ifo, test_ifo, verbose, start_time)

    # main optimization loop
    for epoch in range(num_passes):
        start_time = time.time()

        # check termination
        if termination_criterion is not None and termination_criterion(model, train_ifo):
            if verbose:
                print('Exiting after {} epochs because termination criterion has been met.'.format(epoch))
            break

        # loop over tests
        for i in range(len(train_ifo)):
            idx = rng.randint(0, len(train_ifo))
            w_s, l_s = train_ifo.linear_minimization_oracle(model, idx)  # Alg. 4 of paper
            w_s *= 1.0 / (strong_convexity * len(train_ifo))
            l_s /= len(train_ifo)

            # step length
            temp1 = model.dot(model_pts[idx].T).sum()
            temp2 = model.dot(w_s.T).sum()
            temp3 = np.linalg.norm(model_pts[idx] - w_s) ** 2
            if temp3 > 1e-20:  # no change in iterates, do not perform updates
                gamma = (strong_convexity * (temp1 - temp2) -
                         losses[idx] + l_s) / (strong_convexity * temp3)
                gamma = min(1.0, max(gamma, 0.0))  # clip
                # update
                w_pts_new = (1 - gamma) * model_pts[idx] + gamma * w_s
                loss_new = (1 - gamma) * losses[idx] + gamma * l_s
                model += w_pts_new - model_pts[idx]
                model_pts[idx] = w_pts_new
                losses[idx] = loss_new
            # average
            model_avg *= count / (count + 2)
            model_avg += model * 2.0 / (count + 2)
            count += 1

        # logging
        if logging:
            _log(log_output, epoch+1, model_avg, reg_penalties, train_ifo, dev_ifo, test_ifo, verbose, start_time)

    return model_avg, log_output


# Utility functions

def _get_optimizer(algorithm):
    """Return the appropriate subclass of Optimizer."""
    if algorithm.lower() in ['sgd', 'pegasos', 'sgdoptimizer']:
        return SGDOptimizer
    elif algorithm.lower() in ['svrg', 'svrgoptimizer']:
        return SVRGOptimizer
    elif algorithm.lower() in ['casimirsvrg', 'casimir_svrg', 'csvrg', 'csvrg_lr', 'casimirsvrgoptimizer',
                               'catalystsvrg', 'catalyst_svrg', 'catalystsvrgoptimizer']:
        return CasimirSVRGOptimizer
    else:
        raise ValueError('Unknown optimization algorithm,', algorithm)


def _log(log_output, epoch, model, reg_penalties, train_ifo, dev_ifo, test_ifo, verbose, start_time):
    """Logging to keep track of progress of optimization algorithm."""
    log_header = 'Epoch\t\tFunction'
    log_format = '{:0.2f}\t\t{:0.8f}'
    out = [epoch]
    reg = sum([penalty.function_value(model) for penalty in reg_penalties])
    # return non-smooth function value if smoothing was used (Smoothed IFOs return a tuple)
    out_fn_value = np.asarray(train_ifo.batch_function_value(model)).reshape(-1)[0]
    out.append(out_fn_value + reg)
    for ifo, name in zip([dev_ifo, test_ifo], ['Dev', 'Test']):
        if ifo is not None:
            out_fn_value = ifo.batch_function_value(model)
            eval_value = np.asarray(ifo.evaluation_function(model)).reshape(-1)
            log_header += '\t\t{0}_function\t{0}_evaluation'.format(name)
            log_format += '\t\t{:0.6f}\t' + ('\t{:0.6f}' * eval_value.shape[0])
            out.append(out_fn_value + reg)
            out.extend(eval_value)
    log_header += '\t\tTime'
    log_format += '\t\t{:0.2f}'
    out.append(time.time() - start_time)
    log_output.append(out)
    if verbose:
        if epoch == 0:
            print(log_header)
        print(log_format.format(*out))
        sys.stdout.flush()


def _get_strong_convexity_from_regularization(reg_penalties):
    """Return the strong convexity coefficient from regularization penalties."""
    strong_convexity = 0.0
    for reg in reg_penalties:
        strong_convexity += reg.strong_convexity()
    return strong_convexity


def _get_regularization_gradient(reg_penalties, model):
    """Return gradient of all regularization penalties."""
    return sum([penalty.gradient(model) for penalty in reg_penalties])


def _get_learning_rate_function_sgd(learning_rate_scheme,
                                    strong_convexity,
                                    initial_learning_rate,
                                    learning_rate_time_factor):
    """Return appropriate learning rate function based on string input."""
    if learning_rate_scheme == 'const':
        return lambda t: initial_learning_rate
    elif learning_rate_scheme == 'pegasos':
        assert strong_convexity > 0, 'Pegasos learning rate requires non-zero strong convexity.'
        return lambda t: 1 / (strong_convexity * (t + 1))
    elif learning_rate_scheme == 'linear':
        return lambda t: initial_learning_rate / (1 + t / learning_rate_time_factor)
    else:
        raise ValueError('Unknown learning rate scheme for SGDOptimizer,', learning_rate_scheme)


def _get_smoothing_update_function_casimir(smoothing_coefficient_update_rule,
                                           initial_smoothing_coefficient,
                                           strong_convexity):
    """Return function that adapts smoothing coefficient of CasimirSVRGOptimizer based on string input."""
    if initial_smoothing_coefficient is None:
        # no smoothing
        return lambda t, kappa: None
    else:
        # add smoothing
        if smoothing_coefficient_update_rule == 'const':
            return lambda t, kappa: initial_smoothing_coefficient
        elif smoothing_coefficient_update_rule == 'linear':
            return lambda t, kappa: initial_smoothing_coefficient / (t+1)
        elif smoothing_coefficient_update_rule == 'expo':
            def smoothing_update_fun(t, kappa):
                q = kappa / (kappa + strong_convexity)
                factor = math.sqrt(1 - math.sqrt(q) / 2)
                return initial_smoothing_coefficient * (factor ** t)
            return smoothing_update_fun
        else:
            raise ValueError('Unknown smoothing update rule: {}'.format(smoothing_coefficient_update_rule))


def _get_moreau_coefficient_update_function_casimir(moreau_coefficient_update_rule,
                                                    initial_moreau_coefficient):
    """Return function to update Moreau coefficient (weight of prox term) of CasimirSVRGOptimizer from string input."""
    if moreau_coefficient_update_rule == 'const':
        return lambda t: initial_moreau_coefficient
    elif moreau_coefficient_update_rule == 'linear':
        return lambda t: (t+1) * initial_moreau_coefficient
    else:
        raise ValueError('Unknown Moreau coefficient update rule: {}'.format(moreau_coefficient_update_rule))
