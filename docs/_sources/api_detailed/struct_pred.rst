API reference for structured_prediction_experiment
==================================================

.. autofunction:: casimir.structured_prediction_experiment.utils.make_parser
.. autofunction:: casimir.structured_prediction_experiment.utils.get_output_filename
.. autofunction:: casimir.structured_prediction_experiment.utils.get_ifo
.. autofunction:: casimir.structured_prediction_experiment.utils.run_algorithm


Command line arguments
----------------------
The table below describes command line arguments for the structured prediction experiments,
namely named entity recognition and visual object localization. The main files are respectively
``main_ner.py`` and ``main_loc.py``.
The common arguments apply to both these files, which specialized arguments apply only to individual files.
This is followed by arguments unique to each individual main file.

.. csv-table:: **Common command line arguments**
	:widths: 15 10 10 65
	:header-rows: 1

	Argument Name,	Type,	Default,	Description
	l2reg,	float,	0.1,	:math:`\ell_2` regularizer which taken to be ``l2reg``:math:`/n` where :math:`n` is the number of examples
	prefix,	str,	``'./data'``,	Prefix applied to all data paths such as ``train_file`` or ``bbox-dir`` below
	output-dir,	str, ``'./outputs'``, 	Directory where logged outputs are saved
	algorithm, str, ``'sgd'``, 	Which optimization algorithm to use. See below for a detailed description_ of options and what they imply.
	num_passes, int, 100,	Maximum number of passes to make through data
	seed, float, 1234, Random seed to use for reproducibility

	lr, float, 1.0, Learning rate for SGD or SVRG or Casimir-SVRG (``'csvrg_lr'``). Ignored by ``'csvrg'``
	lr-t, float, 1.0, Learning rate decay for SGD and ignored by other algorithms. Use learning rate :math:`\eta_t = \eta_0 / (1 + t/t_0)` where :math:`\eta_0` is specified using ``--lr`` and :math:`t_0` is specified using ``--lr-t``
	L, float, 1.0,	Gradient Lipschtiz coefficient for use by Casimir-SVRG (``'csvrg'``). Ignored by other algorithms

	smoother, float, 1.0, Initial smoothing coefficient for algorithms that require smoothing. Ignored by algorithms that do not require smoothing
	decay_smoother, str, 'none', Type to smoothing decay to be applied. Options are ``'none'`` for no decay or ``'expo'`` for :math:`\mu_0 c_t^{-t}` or ``'linear'`` for :math:`\mu_0 / t` where :math:`\mu_0` is specified via ``--smoother``. Refer to documentation of ``casimir.optim.CasimirSVRGOptimizer`` for details.
	K, int, 5, the value of :math:`K` for top-:math:`K` inference for :math:`\ell_2` smoothing

	warm_start, int, 3, Warm start strategy for Casimir-SVRG. ``1`` refers to ``prox-center`` while ``2`` to ``extrapolation`` and ``3`` to ``prev-iterate``. Refer to documentation of ``casimir.optim.CasimirSVRGOptimizer`` for details.
	kappa, float, 1.0, scaling factor on initial Moreau coefficient for Casimir-SVRG variant ``'csvrg_lr'``
	





.. csv-table:: **Command line arguments specific to** ``main_ner.py``
	:widths: 10 10 10 70
	:header-rows: 1

	Argument Name,	Type,	Default,	Description
	train_file, str, ``'conll03_ner/eng.train'``, Name of text file with training data
	dev_file, str, ``'conll03_ner/eng.testa'``, Name of text file with development data
	test_file, str, ``'conll03_ner/eng.testb'``, Name of text file with †esting data

.. csv-table:: **Command line arguments specific to** ``main_loc.py``
	:widths: 10 10 10 70
	:header-rows: 1

	Argument Name,	Type,	Default,	Description
	object-class, str, ``'dog'``, Name of object class to run experiment on
	bbox-dir, str, ``'voc2007/bboxes'``, Name of directory with bounding box information. See `this page <../expt.html>`_ on how to obtain this data
	features-dir, str, ``'voc2007/features'``, Name of directory with bounding box features. See `this page <../expt.html>`_ on how to obtain this data


..  _description:
.. csv-table:: **Description of algorithms accepted by** ``--algorithm``
	:header-rows: 1
	:widths: 10 90

	Algorithm,	Description
	``'csvrg'``, Casimir-SVRG with learning rate specified implicitly via ``grad_lipshitz_parameter``. Specify command line argument ``--L`` for ``grad_lipschitz_parameter``. Requires smoothing.
	``'csvrg_lr'``, Casimir-SVRG with learning rate specified direcly as ``--lr``. Requires smoothing.
	``'svrg'``, SVRG with learning rate specified via ``--lr``. Requires smoothing.
	``'sgd'``, SGD with learning rates :math:`\eta_t = \eta_0 / (1 + t/t_0)` where :math:`\eta_0` is specified using ``--lr`` and :math:`t_0` is specified using ``--lr-t`` and weighted averaging :math:`\bar w_T = \frac{2}{T(T+1)} \sum_{t=0}^{T} t \cdot w_t`
	``'sgd_const'``, SGD with constant learning rate specified as :math:`\eta_0` and weighted averaging :math:`\bar w_T = \frac{2}{T(T+1)} \sum_{t=0}^{T} t \cdot w_t`
	``'pegasos'``, SGD with learning rates :math:`\eta_t = (\lambda t)^{-1}` where :math:`\lambda` is the :math:`L_2` regularization and weighted averaging :math:`\bar w_T = \frac{2}{T(T+1)} \sum_{t=0}^{T} t \cdot w_t`
	``'bcfw'``, Block Coordinate Frank Wolfe optimization for Structural SVMs.
