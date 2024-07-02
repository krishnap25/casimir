Getting Started
===============

Prerequisites
-------------

The optimization algorithms are based on `Numpy <http://www.numpy.org/>`_.
If using ``conda``, run the following command to install all necessary packages and then activate the environment::

   $ conda env create --file environment.yml
   $ source activate casimir

If not using ``conda``, the file environment.yml contains the names of the required packages.

Installation
------------

Clone the repository available here::

   $ git clone https://github.com/krishnap25/casimir.git
   $ cd casimir/

The following command complies the `Cython <http://cython.org/>`_ code, which is needed only for the
experiments on named entity recognition::

   $ ./scripts/compile_cython.sh


Quick start: Binary Classification
----------------------------------

Here is an example on how to use this code base for binary classification on the Iris dataset (class 1 or not class 1).
Run the following in Python REPL ::

   >>> import sklearn.datasets as ds
   >>> from casimir.data.classification import LogisticRegressionIfo
   >>> X, y = ds.load_iris(return_X_y=True)
   >>> ifo = LogisticRegressionIfo(X, y==1)

This objective function can then be optimized using ``casimir.optim.optimize_ifo``, using different
optimization algorithms such as Casimir-SVRG, SVRG or SGD.
In this example, we run 20 passes of SGD with a constant learning rate of 0.05 and weighted averaging (default):

    >>> import numpy as np
    >>> import casimir.optim as optim
    >>> w, logs = optim.optimize_ifo(np.zeros(4), ifo, algorithm='SGD', num_passes=20,
                                     optim_options={'initial_learning_rate': 0.05})

This package prints out the function value after each pass through the dataset.


**Playing with a larger dataset**

Let us now experiment with a larger dataset.
Download the covtype dataset (size 53.8 MB) from `this link <https://drive.google.com/open?id=1SYQWnW1elEq5QqzAe0rA8bqqAdmm82-m>`_
and place it in the folder ``data/``. Then, run the file ``examples/logistic_regression.py`` as follows:

    >>> python examples/logistic_regression.py

The data is normalized so that a learning rate of 1 will work for SVRG and SGD. Feel free to play around with the
optimization options in CasimirSVRG, SVRG and SGD. Note that the optimization algorithm is controlled by
the parameter ``algorithm`` of ``optim.optimize_ifo``.
For reference, with the given parameter settings, at the end of ten iterations, Casimir-SVRG
should achieve a function value of 0.6624 with ``warm_start = 'prox-center'`` and 0.6606 with ``warm_start = 'extrapolation'``
while simple SVRG reaches a function value of 0.0664.


Starting with Named Entity Recognition
--------------------------------------

If you already have the CoNLL-2003 dataset for named entity recognition and have installed the Cython code above, then
proceed to ``examples/named_entity_recognition.py``. If not, consult `this page <./expt.html>`_ on how to obtain the data.
This example is structured in much the same way as the previous example. It assumes that the data is available
in the folder ``data/conll03_ner/``.
A learning rate of about :math:`10^{-2}` works for SGD and SVRG (more aggressive learning rates work for SVRG as well). For Casimir-SVRG, try
setting ``grad_lipschitz_parameter`` to 100 or thereabouts.

    Note: These learning rates mentioned above are not tuned for best performance, but are simply ballpark numbers
    to get started. The parameters obtained from tuning may be found in ``scripts/named_entity_recognition.sh``.

