from __future__ import absolute_import, division, print_function
import numpy as np
import os
import sys
import scipy.io as sio
sys.path.append(os.path.abspath('./'))
import casimir.data.classification as classification
import casimir.optim as optim


# Get data from this link: https://drive.google.com/open?id=1SYQWnW1elEq5QqzAe0rA8bqqAdmm82-m
# modify path of data_file if necessary
data_file = './data/covtype.full.mat'
assert os.path.isfile(data_file)

# Step 1.1: load data from mat file
arrays = sio.loadmat(data_file)
X = np.array(arrays['X']).T
y = np.array(arrays['y'], dtype=np.int)[:, 0] - 1
n = y.shape[0]

# Step 1.2: Create IFO. Similar to the example of iris in the documentation
ifo = classification.LogisticRegressionIfo(X, y)
initial_model = np.zeros(54)  # this dataset is 54-dimensional


# Step 2: Set optimization parameters
l2reg = 0.01 / n  # ill-conditioned problem, Casimir-SVRG will be faster because of acceleration
# Since we are already in the smooth case, Casimir-SVRG and Catalyst-SVRG are identical here.
grad_lipschitz_parameter = 0.25  # data is normalized so this bound works for the logistic loss
learning_rate = 1 / (0.25 + l2reg)  # aggressive learning rate works for SVRG and Casimir-SVRG
initial_learning_rate = 1.0  # smaller learning rate for SGD
print('learning rate:', learning_rate, 'strong convexity:', l2reg)

l2penalty = optim.L2Penalty(l2reg)  # create L2Penalty object

# Step 3: Set options and run optimization. Change the options below to try different optimization algorithms.

algorithm = 'casimir_svrg'  # Try 'svrg' and 'sgd' as well
# algorithm = 'svrg'
warm_start = 'prox-center'  # Try 'prox-center' and 'extrapolation' to test Casimir warm start schemes


# Step 3.1: set optimization options
if algorithm == 'sgd':
    optim_options = {'initial_learning_rate': initial_learning_rate}

elif algorithm == 'svrg':
    optim_options = {'learning_rate': learning_rate}

else:
    optim_options = {'grad_lipschitz_parameter': grad_lipschitz_parameter,
                     'warm_start': warm_start}

# Step 3.2: Run optimization. Change the parameter ``algorithm`` to try out different optimization algorithms.
model, logs = optim.optimize_ifo(initial_model, ifo, algorithm=algorithm, dev_ifo=None, test_ifo=None,
                                 reg_penalties=[l2penalty], num_passes=10, termination_criterion=None, seed=25,
                                 logging=True, verbose=True,
                                 optim_options=optim_options)

# For reference, with the given parameter settings, at the end of ten iterations, Casimir-SVRG
# should achieve a function value of 0.6624 with warm_start = 'prox-center' and
# 0.6606 with warm_start = 'extrapolation'
# while simple SVRG reaches a function value of 0.0664.
# The optimal function value for this setting is 6.59602842590e-01.
