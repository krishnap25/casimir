from __future__ import absolute_import, division, print_function
import numpy as np
import casimir.optim as optim


class LogisticRegressionIfo(optim.IncrementalFirstOrderOracle):
    """Create an incremental First Order Oracle for logistic regression.

        The component function :math:`f_i` is the logistic loss defined on the :math:`i` th example
        for binary classficiation.

        :param data: matrix of size :math:`n \\times d`, where :math:`n` is the number of data points
            and :math:`d` is dimensionality of each data point
        :param labels: vector of size :math:`n`, with entries 0 or 1
    """
    def __init__(self, data, labels):
        super(LogisticRegressionIfo, self).__init__()
        self.data = data
        self.labels = labels

    def function_value(self, model, idx):
        pred = np.matmul(self.data[idx:idx+1, :], model)
        target = self.labels[idx:idx+1]
        return _logistic_loss(pred, target)

    def gradient(self, model, idx):
        return _logistic_loss_gradient(self.data[idx:idx + 1, :], self.labels[idx:idx + 1], model)

    def batch_function_value(self, model):
        score = np.matmul(self.data, model)
        target = self.labels
        return _logistic_loss(score, target)

    def batch_gradient(self, model):
        return _logistic_loss_gradient(self.data, self.labels, model)

    def evaluation_function(self, model):
        """Compute the classification error (0-1 loss)."""
        predictions = (1 + np.sign(np.matmul(self.data, model))) / 2.0
        return (predictions - self.labels).abs().sum() / self.labels.shape[0]

    def __len__(self):
        return self.labels.shape[0]


def _logistic_loss(score, target):
    probs = 1 / (1 + np.exp(-score))
    return -(target * np.log(probs) + (1 - target) * np.log(1 - probs)).sum() / target.shape[0]


def _logistic_loss_derivative(score, target):
    return (-target / (1 + np.exp(score)) + (1 - target) / (1 + np.exp(-score))) / target.shape[0]


def _logistic_loss_gradient(data, labels, model):
    return np.matmul(_logistic_loss_derivative(np.matmul(data, model), labels), data)
