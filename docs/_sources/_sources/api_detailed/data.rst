API reference for casimir.data
=============================================
This page summarizes the API for various data related operations for tasking including binary classification,
named entity recognition and object localization. Lastly, it talks about how to use the optimization algorithms to
new tasks or datasets.

.. _classification:

Classification
--------------

.. autoclass:: casimir.data.LogisticRegressionIfo
    :members:


.. _named entity recognition:

Named Entity Recognition:
--------------------------

.. autofunction:: casimir.data.named_entity_recognition.create_ner_ifo_from_data
.. autoclass:: casimir.data.named_entity_recognition.NamedEntityRecognitionIfo
    :members:
.. autoclass:: casimir.data.named_entity_recognition.NerDataset
    :members:
.. autofunction:: casimir.data.named_entity_recognition.viterbi_decode
.. autofunction:: casimir.data.named_entity_recognition.viterbi_decode_top_k


.. _visual object localization:

Visual Object Localization
--------------------------

.. autofunction:: casimir.data.localization.create_loc_ifo_from_data
.. autoclass:: casimir.data.localization.LocalizationIfo
    :members:
.. autoclass:: casimir.data.localization.VocDataset
    :members:


Extending to new tasks and datasets
------------------------------------
The framework of IFOs decouples the optimization from the data and loss function used, as captured by the figure below.

.. image:: ../fig/fig/fig.001.png
    :scale: 50 %
    :align: center

In order to define a new incremental first order oracle, one must override the class
``casimir.optim.IncrementalFirstOrderOracle`` or the class
``casimir.optim.SmoothedIncrementalFirstOrderOracle``.
See the `documentation of these classes <optim.html>`_ for more details.

..
    See IFOs for logistic regression (`casimir/data/classification.py`) or
    for structural support vector machines for named entity recognition
    (`casimir/data/named_entity_recognition/`)
    and visual object localization (`casimir/data/localization/`) for examples on how this is done.

See IFOs for classification_ or for structural support vector machines for `named entity recognition`_ and
`visual object localization`_ for reference on how this is done.

