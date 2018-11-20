from __future__ import absolute_import, division, print_function
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('./'))
import casimir.data.named_entity_recognition as ner  # noqa: E402
import casimir.optim as optim  # noqa: E402


# Make sure that these files exist to run this example. Otherwise, edit file names below to reflect the true names:
train_file = 'data/ner/eng.train'
dev_file = 'data/ner/eng.testa'
test_file = 'data/ner/eng.testb'

assert os.path.isfile(train_file), 'train_file {} does not exist'.format(train_file)
assert os.path.isfile(dev_file), 'dev_file {} does not exist'.format(dev_file)
assert os.path.isfile(test_file), 'test_file {} does not exist'.format(test_file)

# Step 1: Create IFO from input files
train_ifo, dev_ifo, test_ifo = ner.create_ner_ifo_from_data(train_file, dev_file, test_file,
                                                            smoothing_coefficient=10.0, num_highest_scores=5)

# Step 2: Set optimization parameters
l2reg = 0.1 / len(train_ifo)
print('l2reg:', l2reg)
l2penalty = optim.L2Penalty(l2reg)

dim = 2**16 - 1  # hard-coded. All features are hashed onto these dimensions.
model = np.zeros(dim)

# Step 3: Set optim_options and run optimization

# Casimir-SVRG, constant smoothing
optim_options_1 = {'grad_lipschitz_parameter': 32.0, 'initial_smoothing_coefficient': 10.0,
                   'warm_start': 'prev-iterate'}

# Casimir-SVRG, decaying smoothing
optim_options_2 = {'learning_rate': 2e-2, 'initial_smoothing_coefficient': 2.0, 'initial_moreau_coefficient': l2reg,
                   'warm_start': 'extrapolation'}

num_passes = 10

# Run optimization
model, logs = optim.optimize_ifo(model, train_ifo, algorithm='casimir_svrg', dev_ifo=dev_ifo, test_ifo=None,
                                 reg_penalties=[l2penalty], num_passes=num_passes, termination_criterion=None, seed=25,
                                 logging=True, verbose=True,
                                 optim_options=optim_options_2)


print(logs)

# After ten iterations, SVRG achieves a loss of about 1.3 with learning rate 2e-2.
# Casimir-SVRG achieves a loss of close to 1.2 with either optim_options_1 or optim_options_2 above.

# For sample parameter values for other optimization algorithms, consult scripts/named_entity_recognition.sh.
