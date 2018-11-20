"""
.. module:: incremental_first_order_oracle
   :synopsis: Module with definitions of abstract incremental first order oracle class and its smoothed version.


.. moduleauthor:: Krishna Pillutla <last-name AT cs DOT washington DOT edu>

"""
from __future__ import absolute_import, division, print_function


class IncrementalFirstOrderOracle(object):
    """ Base class for incremental first order oracles (IFO).

        For a function :math:`f(w)` defined as :math:`f(w) = \\frac{1}{n} \\sum_{i=1}^n f_i(w)`,
        this class is an interface to implement methods to compute
        :math:`f_i(w)` and its gradient :math:`\\nabla f_i(w)`,
        as well as the batch function value :math:`f(w)` and the batch gradient :math:`\\nabla f(w)`
        if each :math:`f_i` is smooth. If :math:`f_i` is not smooth,
        use sub-class :class:`SmoothedIncrementalFirstOrderOracle` instead.

        A concrete implementation must at least override the methods ``function_value``, ``gradient`` and ``__len__``.
        Optionally, it may also override ``evaluation_function`` for domain specific evaluation metrics such as
        classification accuracy. The methods ``batch_function_value`` and ``batch_gradient`` are implemented as a loop
        that averages over the outputs of ``function_value`` and ``gradient`` respectively by default.
        If a more efficient implementation exists, these methods can be overridden by derived classes as well.

        Input ``idx`` to ``function value`` and ``gradient`` must not exceed :math:`n`.
        This is not explicitly checked here.
    """
    def __init__(self):
        pass

    def __len__(self):
        """Return the number of component functions :math:`n`.
        """
        raise NotImplementedError

    def function_value(self, model, idx):
        """Return function value :math:`f_i(w)` where :math:`w` is ``model`` and :math:`i` is ``idx``."""
        raise NotImplementedError

    def gradient(self, model, idx):
        """Return gradient :math:`\\nabla f_i(w)` where :math:`w` is ``model`` and :math:`i` is ``idx``."""
        raise NotImplementedError

    def batch_function_value(self, model):
        """Return function value :math:`f(w)` where :math:`w` is ``model``."""
        return _batch_average(self.function_value, model, len(self))

    def batch_gradient(self, model):
        """Return gradient :math:`\\nabla f(w)` where :math:`w` is ``model``."""
        return _batch_average(self.gradient, model, len(self))

    def evaluation_function(self, model):
        """Return domain-specific task metrics (default ``None``)."""
        return None


class SmoothedIncrementalFirstOrderOracle(IncrementalFirstOrderOracle):
    """ Base class of smoothed incremental first order oracles.

        For a function :math:`f(x)` defined as :math:`f(w) = \\frac{1}{n} \\sum_{i=1}^n f_i(w)`,
        where each :math:`f_i` is non-smooth but smoothable,
        this class is an interface to implement methods to compute
        :math:`f_{i, \mu}(w)` and its gradient :math:`\\nabla f_{i, \mu}(w)`,
        as well as the batch function value :math:`(f(w), f_{\mu}(w))`
        and the batch gradient :math:`\\nabla f_{\mu}(w)`.
        Here, :math:`g_\mu` is a smooth surrogate to the non-smooth function :math:`g`
        that is parameterized by a smoothing coefficient :math:`\mu`.

        When :math:`\mu` is ``None``(i.e., no smoothing), the implementation must serve as an IFO for
        the original non-smooth function :math:`f`,
        in which case :math:`\\nabla f_i` refers to a subgradient of :math:`f_i`.

        .. note::
            This class contains a field ``smoothing_coefficient`` to represent :math:`\mu`,
            which can be mutated by optimization algorithms or other functions that use adaptive smoothing schemes.
    """
    def __init__(self, smoothing_coefficient=None):
        super(SmoothedIncrementalFirstOrderOracle, self).__init__()
        assert smoothing_coefficient is None or smoothing_coefficient > 0
        self.smoothing_coefficient = smoothing_coefficient

    def update_smoothing_coefficient(self, smoothing_coefficient):
        assert smoothing_coefficient is None or smoothing_coefficient > 0
        self.smoothing_coefficient = smoothing_coefficient

    def __len__(self):
        raise NotImplementedError

    def function_value(self, model, idx):
        """Return the pair :math:`\\big( f_i(w), f_{i, \mu}(w) \\big)`.
            If ``smoothing_coefficient`` is ``None``, i.e., no smoothing,
            simply return :math:`f_i(w)`, where :math:`i` represents the index ``idx``.
        """
        raise NotImplementedError

    def gradient(self, model, idx):
        """ Return that gradient :math:`\\nabla f_{i, \mu}(w)` if ``smoothing_coefficient`` is not ``None``
            or :math:`\\nabla f_i(w)` if ``smoothing_coefficient`` is ``None``.
        """
        raise NotImplementedError


def _batch_average(func, argument, length):
    """Computes average of a function func(argument, .) over a batch [0, 1, ..., length-1]."""
    to_return = func(argument, 0)
    for i in range(1, length):
        to_return += func(argument, i)
    to_return /= length
    return to_return
