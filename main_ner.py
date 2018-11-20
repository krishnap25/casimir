from __future__ import absolute_import, division, print_function
import numpy as np
import pickle as pkl
import casimir.structured_prediction_experiment.utils as utils

if __name__ == '__main__':
    task = 'ner'
    parser = utils.make_parser(task)
    args = parser.parse_args()
    output_name = utils.get_output_filename(args, task)

    train_ifo, dev_ifo, test_ifo = utils.get_ifo(args, task)

    dim = 2**16 - 1
    initial_model = np.zeros(dim)
    model, logs = utils.run_algorithm(args, initial_model, train_ifo, dev_ifo, test_ifo)

    with open(output_name, 'wb') as fn:
        pkl.dump([logs, args], fn, protocol=2)
    print('Done Training')

