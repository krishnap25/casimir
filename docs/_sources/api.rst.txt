Casimir API summary
==================================

.. currentmodule:: casimir

Optimization
------------

Incremental First Order Oracles - Base classes:

.. autosummary::

    casimir.optim.IncrementalFirstOrderOracle
    casimir.optim.SmoothedIncrementalFirstOrderOracle

Optimization Algorithms:

.. autosummary::

    casimir.optim.optimize_ifo
    casimir.optim.CasimirSVRGOptimizer
    casimir.optim.SGDOptimizer
    casimir.optim.SVRGOptimizer
    casimir.optim.block_coordinate_frank_wolfe_optimize

Regularization:

.. autosummary::

    casimir.optim.L2Penalty

Data
----

Classification:

.. autosummary::

    casimir.data.LogisticRegressionIfo

Named Entity Recognition:

.. autosummary::

    casimir.data.named_entity_recognition.create_ner_ifo_from_data
    casimir.data.named_entity_recognition.NamedEntityRecognitionIfo
    casimir.data.named_entity_recognition.NerDataset
    casimir.data.named_entity_recognition.viterbi_decode
    casimir.data.named_entity_recognition.viterbi_decode_top_k


Object Localization:

.. autosummary::

    casimir.data.localization.create_loc_ifo_from_data
    casimir.data.localization.LocalizationIfo
    casimir.data.localization.VocDataset

Structured Prediction Utilities
-------------------------------

.. autosummary::

    casimir.structured_prediction_experiment.utils.make_parser
    casimir.structured_prediction_experiment.utils.get_output_filename
    casimir.structured_prediction_experiment.utils.get_ifo
    casimir.structured_prediction_experiment.utils.run_algorithm
