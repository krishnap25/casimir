from __future__ import absolute_import, division, print_function
import argparse
import os
import sys
import casimir.optim as optim
import casimir.data.named_entity_recognition as ner
import casimir.data.localization as loc


def make_parser(task):
    """Create argument parser for structured prediction experiments."""

    parser = argparse.ArgumentParser(description='Arguments for smooth structured prediction')
    if task == 'loc':
        parser.add_argument('--object-class', type=str, default='dog',
                            help='object class to use, for localization task')
    parser.add_argument('--l2reg', type=float, default=0.1, help='l2 regularizer, taken to be l2reg/n')
    parser.add_argument('--algorithm', type=str,
                        choices=['sgd', 'sgd_const', 'csvrg', 'csvrg_lr',
                                 'svrg', 'pegasos', 'bcfw'],
                        default='sgd', help='optimization algorithm')

    if task == 'ner':
        parser.add_argument('--small', help='use fraction of dataset only',
                            action='store_true')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate for sgd, svrg, csvrg_lr')
    parser.add_argument('--lr-t', type=float, default=1.0, help='learning rate decay for sgd: lr / (1 + t / lr_t)')
    parser.add_argument('--L', type=float, default=1.0, help='Gradient Lipschitz' +
                                                             'coefficient of function. Required for Casimir-SVRG')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--smoother', type=float, help='smoothing coefficient')
    parser.add_argument('--K', type=int, default=5, help='K for top-K strategy')
    parser.add_argument('--num_passes', type=int, default=50, help='number of passes through the data')
    parser.add_argument('--warm_start', help='Casimir warm start rule. Choose 1, 2 or 3', type=int, default=3)
    parser.add_argument('--kappa', type=float, help='Scaling factor on initial Moreau coefficient' +
                        'for the adaptive smoothing scheme', default=1.0)
    parser.add_argument('--decay_smoother', help='decay smoothing parameter? Say "const" for no decay of smoothing\
                "expo" for exp(-t) and "linear" for 1/t', type=str, default='const',
                        choices=['none', 'linear', 'expo'])

    # Data directories
    parser.add_argument('--prefix', type=str, default='./data/', help='Prefix applied to all files and folders.')
    if task == 'ner':
        parser.add_argument('--train-file', type=str, default='conll03_ner/eng.train')
        parser.add_argument('--dev-file', type=str, default='conll03_ner/eng.testa')
        parser.add_argument('--test-file', type=str, default='conll03_ner/eng.testb')
    elif task == 'loc':
        parser.add_argument('--bbox-dir', type=str, default='voc2007/bboxes')
        parser.add_argument('--features-dir', type=str, default='voc2007/features')
    parser.add_argument('--output-dir', type=str, default='outputs')

    return parser


def get_output_filename(args, task, size=''):
    """Get name of output file based on arguments and task."""
    # output directory is output/ner/... or output/loc/{obj_class}/...
    smooth = (args.smoother is not None and args.smoother > 0)

    if args.algorithm in ['sgd_const', 'svrg', 'csvrg_lr']:
        lr_or_l = args.lr
    elif args.algorithm == 'sgd':
        lr_or_l = '{0}_{1}'.format(args.lr, args.lr_t)
    else:
        lr_or_l = args.L

    if args.decay_smoother != 'none':
        alg_name = '{0}_{1}'.format(args.algorithm, args.decay_smoother)
    else:
        alg_name = args.algorithm

    task_name = 'ner' if task == 'ner' else '{}/{}'.format(task, args.object_class)
    prefix = os.path.join(args.output_dir, task_name)

    if not smooth:
        output_name = '{0}_{1}_{2}_{3}_{4}.p'.format(
            args.algorithm, args.l2reg, lr_or_l, size, args.seed)
    else:
        output_name = '{0}_{1}_{2}_{3}_{4}_{5}_{6}.p'.format(
            alg_name, args.l2reg, lr_or_l, size, args.seed, args.smoother, args.K)
        if args.algorithm in ['csvrg', 'csvrg_lr']:
            output_name = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_option{7}.p".format(
                alg_name, args.l2reg, lr_or_l, size, args.seed, args.smoother, args.K, args.warm_start)
    return prefix + output_name


def get_ifo(args, task):
    """Get appropriate Incremental First Order Oracle (IFO) for structured prediction task."""
    if task == 'ner':
        train_ifo, dev_ifo, test_ifo = ner.create_ner_ifo_from_data(os.path.join(args.prefix, args.train_file),
                                                                    os.path.join(args.prefix, args.dev_file),
                                                                    os.path.join(args.prefix, args.test_file),
                                                                    smoothing_coefficient=args.smoother,
                                                                    num_highest_scores=args.K)
    elif task == 'loc':
        train_ifo, dev_ifo, test_ifo = loc.create_loc_ifo_from_data(args.object_class,
                                                                    os.path.join(args.prefix,
                                                                                 args.bbox_dir,
                                                                                 '{}.p'.format(args.object_class)),
                                                                    os.path.join(args.prefix, args.features_dir),
                                                                    smoothing_coefficient=args.smoother,
                                                                    num_highest_scores=args.K)
    else:
        raise ValueError('Unknown task:', task)
    return train_ifo, dev_ifo, test_ifo


warm_start_dict = {1: 'prox-center', 2: 'extrapolation', 3: 'prev-iterate'}


def run_algorithm(args, initial_model, train_ifo, dev_ifo, test_ifo):
    """Run optimization algorithm for structured prediction experiments."""

    l2reg = args.l2reg / len(train_ifo)
    reg_penalties = [optim.L2Penalty(l2reg)]

    print('l2reg =', l2reg)
    print('Starting optimization with', args.algorithm)
    sys.stdout.flush()

    if args.algorithm == 'bcfw':
        # not in incremental first-order oracle model, so handle separately and return
        model, log_output = optim.block_coordinate_frank_wolfe_optimize(initial_model, train_ifo, dev_ifo, test_ifo,
                                                                        reg_penalties, num_passes=args.num_passes,
                                                                        seed=args.seed, logging=True, verbose=True)

        return model, log_output

    # IFO algorithms
    if args.algorithm in ['sgd', 'sgd_const', 'pegasos']:
        if args.algorithm == 'sgd':
            lr_scheme = 'linear'
        elif args.algorithm == 'sgd_const':
            lr_scheme = 'const'
        else:
            lr_scheme = 'pegasos'
        optim_options = {
            'learning_rate_scheme': lr_scheme,
            'initial_learning_rate': args.lr,
            'learning_rate_time_factor': args.lr_t,
            'averaging': 'wavg'
        }
        algorithm = 'sgd'

    elif args.algorithm == 'svrg':
        optim_options = {
            'learning_rate': args.lr,
            'smoothing_coefficient': args.smoother
        }
        algorithm = 'svrg'
    elif args.algorithm == 'csvrg':
        optim_options = {
            'grad_lipschitz_parameter': args.L,
            'initial_smoothing_coefficient': args.smoother,
            'smoothing_coefficient_update_rule': args.decay_smoother,
            'warm_start': warm_start_dict[args.warm_start]
        }
        algorithm = 'casimir_svrg'
    elif args.algorithm == 'csvrg_lr':
        optim_options = {
            'learning_rate': args.lr,
            'initial_smoothing_coefficient': args.smoother,
            'smoothing_coefficient_update_rule': args.decay_smoother,
            'warm_start': warm_start_dict[args.warm_start],
            'initial_moreau_coefficient': l2reg * args.kappa
        }
        algorithm = 'casimir_svrg'
    else:
        raise NotImplementedError

    model, log_output = optim.optimize_ifo(initial_model, train_ifo, dev_ifo, test_ifo, algorithm,
                                           reg_penalties, num_passes=args.num_passes, termination_criterion=None,
                                           seed=args.seed, logging=True, verbose=True, optim_options=optim_options)

    return model, log_output
