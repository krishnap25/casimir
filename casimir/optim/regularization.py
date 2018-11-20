"""
.. module:: optimization_algorithms
   :synopsis: Define regularization penalties for use.


.. moduleauthor:: Krishna Pillutla <last-name AT cs DOT washington DOT edu>

"""
from __future__ import absolute_import, division, print_function
import numpy as np


class SmoothRegularizationPenalty(object):
    """Base class for smooth, convex regularization penalties."""
    def function_value(self, model):
        """Return function value of regularization penalty."""

    def gradient(self, model):
        """Return gradient of penalty at model."""
        raise NotImplementedError

    def strong_convexity(self):
        """Strong convexity coefficient w.r.t. L2 norm."""
        return 0


class ProximalRegularizationPenalty(object):
    """Base class for (potentially non-smooth) convex regularization penalties with a tractable prox operation."""
    def function_value(self, model):
        """Return function value of regularization penalty."""

    def prox(self, model):
        """Return prox operated upon model."""
        raise NotImplementedError

    def prox_(self, model):
        """Apply prox operator in-place."""
        raise NotImplementedError


class L2Penalty(SmoothRegularizationPenalty):
    """Class representing the L2 regularization penalty.

        For a prox-center :math:`z` and regularization parameter :math:`\lambda`,
        the penalty takes the form :math:`r(w) = \\frac{\lambda}{2}\|w-z\|^2_2`.

        :param regularization_parameter: the parameter :math:`\lambda` above.
        :param prox_center: Ndarray representing :math:`z` above.
                A value of ``None`` is interpreted as the zero vector.
    """
    def __init__(self, regularization_parameter, prox_center=None):
        self.regularization_parameter = regularization_parameter
        self.prox_center = prox_center

    def function_value(self, model):
        if self.prox_center is None:
            return 0.5 * self.regularization_parameter * model.dot(model)
        else:
            return 0.5 * self.regularization_parameter * np.linalg.norm(model - self.prox_center) ** 2

    def gradient(self, model):
        if self.prox_center is None:
            return self.regularization_parameter * model
        else:
            return self.regularization_parameter * (model - self.prox_center)

    def strong_convexity(self):
        return self.regularization_parameter
