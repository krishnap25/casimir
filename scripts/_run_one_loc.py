from __future__ import absolute_import, division, print_function
from collections import defaultdict
import numpy as np
import os
import pickle as pkl
import sys
sys.path.append(os.path.abspath('./'))
import casimir.structured_prediction_experiment.utils as utils  # noqa: E402

# Script to save on feature loading time for localization experiments

l2reg = 10

if len(sys.argv) > 1:
    obj_class = sys.argv[1]
else:
    raise Exception("Please provide object class as first argument")

parser = utils.make_parser(task='loc')
dim = 2304  # dimensionality of features
num_iter_smooth = 15
num_iter_nonsmooth = 30

# Change directories below
prefix = '/mnt/ssd/'

# Compile list of all tasks
tasks = []
common_smooth = ('--prefix {} --object-class {} --num_passes {} --K 10 ' +
                 '--warm_start 1').format(prefix, obj_class, num_iter_smooth)
common_ns = '--prefix {} --object-class {} --num_passes {}'.format(prefix, obj_class, num_iter_nonsmooth)
SEEDS = list(range(1, 10))

# Choice of hyper-parameters
# csvrg
Ls_csvrg = {'aeroplane': 4096.0,  'bicycle': 4096.0, 'boat': 8192.0, 'bus': 8192, 'car': 4096,
            'cow': 8192.0, 'horse': 8192.0,  'dog': 4096.0, 'pottedplant': 8192.0, 'sheep': 8192.0,
            'person': 32768}
Ls_csvrg = defaultdict(lambda: 2048.0, Ls_csvrg)

smoothers_csvrg = {'boat': 0.1, 'bus': 0.1, 'cow': 0.1, 'horse': 0.1,  'pottedplant': 0.1, 'sheep': 0.1}
smoothers_csvrg = defaultdict(lambda: 1.0, smoothers_csvrg)

# csvrg_lr: with adaptive smoothing
lrs_csvrg = {'car': 0.000244140625, 'tvmonitor': 0.000244140625, 'person': 3.0517578125e-05}
lrs_csvrg = defaultdict(lambda: 0.00048828125, lrs_csvrg)

smoothers_csvrg_lr = defaultdict(lambda: 10.0)

# svrg
lrs_svrg = {'car': 0.000244140625, 'bicycle': 0.0009765625, 'boat': 0.0009765625,
            'horse':  0.0009765625, 'person': 3.0517578125e-05}
smoothers_svrg = {'sheep': 0.1}
lrs_svrg = defaultdict(lambda: 0.00048828125, lrs_svrg)
smoothers_svrg = defaultdict(lambda: 1.0, smoothers_svrg)


# sgd
lrs_sgd = {
     'dog': 0.00048828125,
     'cat': 0.00048828125,
     'bicycle': 0.00048828125,
     'bus': 0.000244140625,
     'aeroplane': 0.0001220703125,
     'tvmonitor': 0.0001220703125,
     'train': 0.0001220703125,
     'bird': 0.000244140625,
     'cow': 0.000244140625,
     'bottle': 6.103515625e-05,
     'diningtable': 0.000244140625,
     'chair': 0.0001220703125,
     'car': 0.0001220703125,
     'sheep': 0.000244140625,
     'boat': 0.00048828125,
     'pottedplant': 0.00048828125,
     'sofa': 0.0001220703125,
     'motorbike': 0.0001220703125,
     'horse': 0.00048828125,
     'person': 6.103515625e-05,
}

t0_sgd = {
     'dog': 1024.0,
     'cat': 4096.0,
     'bicycle': 4096.0,
     'bus': 4096.0,
     'aeroplane': 4096.0,
     'tvmonitor': 4096.0,
     'train': 4096.0,
     'bird': 4096.0,
     'cow': 4096.0,
     'bottle': 4096.0,
     'diningtable': 4096.0,
     'chair': 4096.0,
     'car': 1024.0,
     'sheep': 2048.0,
     'boat': 1024.0,
     'pottedplant': 2048.0,
     'sofa': 4096.0,
     'motorbike': 4096.0,
     'horse': 4096.0,
     'person': 1024.0,
}

# Create arguments strings
# CSVRG
for seed in SEEDS:
    L = Ls_csvrg[obj_class]
    smoother = smoothers_csvrg[obj_class]
    task = ' --algorithm csvrg --l2reg {} --smoother {} --L {} --seed {}'.format(l2reg, smoother, L, seed)
    tasks += [common_smooth + task]

# CSVRG_LR
for seed in SEEDS:
    lr = lrs_csvrg[obj_class]
    smoother = smoothers_csvrg_lr[obj_class]
    task = (' --algorithm csvrg_lr --l2reg {} --smoother {} --lr {} --seed {} ' +
            '--decay_smoother expo').format(l2reg, smoother, lr, seed)
    tasks += [common_smooth + task]

# SVRG
for seed in SEEDS:
    lr = lrs_svrg[obj_class]
    smoother = smoothers_svrg[obj_class]
    task = ' --algorithm svrg --l2reg {} --smoother {} --lr {} --seed {}'.format(l2reg, smoother, lr, seed)
    tasks += [common_smooth + task]

# BCFW
for seed in SEEDS:
    task = ' --algorithm bcfw --l2reg {} --seed {}'.format(l2reg, seed)
    tasks += [common_ns + task]

# Pegasos
for seed in SEEDS:
    task = ' --algorithm pegasos --l2reg {} --seed {}'.format(l2reg, seed)
    tasks += [common_ns + task]

# sgd
for seed in SEEDS:
    lr = lrs_sgd[obj_class]
    t0 = t0_sgd[obj_class]
    task = (' --algorithm sgd --l2reg {}  --lr {} ' + 
            '--lr-t {}  --seed {} ').format(l2reg, lr, t0, seed)
    tasks += [common_ns + task]

# Run
start = False
train_ifo, dev_ifo, test_ifo = None, None, None
for task in tasks:
    args = parser.parse_args(task.split())
    output_name = utils.get_output_filename(args, 'loc')

    if not start:
        start = True
        print('loading data')
        train_ifo, dev_ifo, test_ifo = utils.get_ifo(args, 'loc')

    dim = 2304
    initial_model = np.zeros(dim, dtype=np.float32)
    model, logs = utils.run_algorithm(args, initial_model, train_ifo, dev_ifo, test_ifo)

    with open(output_name, 'wb') as fn:
        pkl.dump([logs, args], fn, protocol=2)
    print('Done Training')
