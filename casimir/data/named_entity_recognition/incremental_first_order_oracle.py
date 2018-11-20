from __future__ import absolute_import, division, print_function
import codecs
import itertools
import numpy as np
from sklearn import metrics
from . import features
from . import reader
from . import viterbi
import casimir.optim.incremental_first_order_oracle as ifo


def create_ner_ifo_from_data(train_file, dev_file=None, test_file=None,
                             smoothing_coefficient=None, num_highest_scores=5):
    """Create Smooth IFOs for train, development and test sets in CoNLL 2003 for the structural hinge loss.

    Each component function :math:`f_i` is the structural hinge loss for the :math:`i` th datapoint :math:`x_i,y_i`.

    :param train_file: Text file with training data.
    :param dev_file: Text file with development data. A value of ``None`` is taken to mean that there is no
        development set.
    :param test_file: Test file with test data. A value of ``None`` is taken to mean that there is no test set.
    :param smoothing_coefficient: Smoothing coefficient to initialize Smoothed IFOs.
        Default ``None`` (i.e., no smoothing).
    :param num_highest_scores: The parameter :math:`K` used by the top-K Viterbi algorithm for inference. Default 5.
    :return: Train, development and test Smoothed IFOs. Dev (test) IFO is ``None`` if dev (test) file is ``None``.
    """
    assert train_file is not None, 'train_file is required'
    with codecs.open(train_file, 'r', 'utf-8') as f:
        train_lines = f.readlines()

    word_to_idx, idx_to_word, _, num_sentences, pos_to_idx, chunk_to_idx, ner_to_idx, idx_to_ner, num_ner_tags = \
        reader.generate_counts_and_feature_map(train_lines, threshold=2)
    train_dataset = reader.NerDataset(
        train_lines, word_to_idx, pos_to_idx, chunk_to_idx, ner_to_idx)

    if dev_file is not None:
        with codecs.open(dev_file, 'r', 'utf-8') as f:
            dev_lines = f.readlines()
        dev_dataset = reader.NerDataset(dev_lines, word_to_idx, pos_to_idx, chunk_to_idx, ner_to_idx)
    else:
        dev_dataset = None

    if test_file is not None:
        with codecs.open(test_file, 'r', 'utf-8') as f:
            test_lines = f.readlines()
        test_dataset = reader.NerDataset(test_lines, word_to_idx, pos_to_idx, chunk_to_idx, ner_to_idx)
    else:
        test_dataset = None

    # Create IFOs. Only train IFO requires smoothing information. Dev and test IFO are used only for evaluation.
    train_ifo = NamedEntityRecognitionIfo(train_dataset, num_ner_tags, ner_to_idx,
                                          smoothing_coefficient, num_highest_scores)
    dev_ifo = NamedEntityRecognitionIfo(dev_dataset, num_ner_tags, ner_to_idx) if dev_file is not None else None
    test_ifo = NamedEntityRecognitionIfo(test_dataset, num_ner_tags, ner_to_idx) if test_file is not None else None
    return train_ifo, dev_ifo, test_ifo


class NamedEntityRecognitionIfo(ifo.SmoothedIncrementalFirstOrderOracle):
    """Create a smoothed incremental first order oracle for structural hinge loss for the task of NER.

    Supports both non-smooth and :math:`\ell_2` smoothed versions of the structural hinge loss
    but not entropy smoothing.

    :param ner_dataset: :class:`NerDataset` object that supports ``__len__`` and ``__getitem__`` methods.
    :param num_ner_tags: Number of NER tags. NER tags are assumed to be 0, 1, ... , ``num_ner_tags``-1.
    :param ner_to_idx: dict mapping each NER tag to its integer ID.
    :param smoothing_coefficient: The amount of smoothing :math:`\mu`.
    :param num_highest_scores: The value K for top-K inference.
    """
    def __init__(self,
                 ner_dataset,
                 num_ner_tags,
                 ner_to_idx,
                 smoothing_coefficient=None,
                 num_highest_scores=5):
        super(NamedEntityRecognitionIfo, self).__init__(smoothing_coefficient)
        self.ner_dataset = ner_dataset
        self.num_ner_tags = num_ner_tags
        self.ner_to_idx = ner_to_idx
        self.num_highest_scores = num_highest_scores

    def __len__(self):
        return len(self.ner_dataset)

    def function_value(self, model, idx):
        words, pos_tags, chunk_tags, ner_tags = self.ner_dataset[idx]
        gold_cost = features.model_dot_features(model, words, pos_tags, chunk_tags, ner_tags)

        if self.smoothing_coefficient is not None:
            # l2 smoothing
            best_scores, best_seqs = viterbi.viterbi_decode_top_k(
                self.num_highest_scores, words, pos_tags, chunk_tags, ner_tags, self.num_ner_tags, model, use_loss=True)

            weights = viterbi.determine_weights_l2(self.smoothing_coefficient, best_scores)
            k = np.count_nonzero(weights)  # only need to compute loss for non-zeros

            # return (non-smooth loss, smooth loss)
            fn_val = np.asarray([best_scores[0] - gold_cost, np.dot(best_scores[:k], weights[:k]) - gold_cost])

        else:
            # no smoothing
            best_score, best_seq = viterbi.viterbi_decode(
                words, pos_tags, chunk_tags, ner_tags, self.num_ner_tags, model, use_loss=True)
            fn_val = best_score - gold_cost

        return fn_val

    def gradient(self, model, idx):
        words, pos_tags, chunk_tags, ner_tags = self.ner_dataset[idx]
        dim = model.shape[0]
        gold_features = features.generate_feature_vector(words, pos_tags, chunk_tags, ner_tags, dim)

        if self.smoothing_coefficient is not None:
            # l2 smoothing
            best_scores, best_seqs = viterbi.viterbi_decode_top_k(
                self.num_highest_scores, words, pos_tags, chunk_tags, ner_tags, self.num_ner_tags, model, use_loss=True)

            weights = viterbi.determine_weights_l2(self.smoothing_coefficient, best_scores)
            k = np.count_nonzero(weights)  # only need to compute gradient for non-zeros

            weighted_sum = sum([features.generate_feature_vector(
                words, pos_tags, chunk_tags, best_seqs[i, :], dim) * weights[i]
                                for i in range(k)])
            grad = weighted_sum - gold_features

        else:
            # no smoothing
            best_score, best_seq = viterbi.viterbi_decode(
                words, pos_tags, chunk_tags, ner_tags, self.num_ner_tags, model, use_loss=True)
            grad = features.generate_feature_vector(words, pos_tags, chunk_tags, best_seq, dim) - gold_features

        return grad

    def linear_minimization_oracle(self, model, idx):
        words, pos_tags, chunk_tags, ner_tags = self.ner_dataset[idx]
        dim = model.shape[0]

        best_score, best_seq = viterbi.viterbi_decode(
            words, pos_tags, chunk_tags, ner_tags, self.num_ner_tags, model, use_loss=True)

        w_s = (features.generate_feature_vector(words, pos_tags, chunk_tags, ner_tags, dim) -
               features.generate_feature_vector(words, pos_tags, chunk_tags, best_seq, dim))

        return w_s, _0_1_loss(ner_tags, best_seq)

    def evaluation_function(self, model):
        """Return the :math:`F_1` score on all tags excluding ``'O'``."""
        eval_tags = list(range(self.num_ner_tags))
        eval_tags.remove(self.ner_to_idx['O'])

        predicted = _predict(model, self.ner_dataset, self.num_ner_tags)
        f1 = metrics.f1_score(_flatten(self.ner_dataset.ner_tags.tolist()),
                              _flatten(predicted), labels=eval_tags,
                              average='weighted')
        return f1


# Helper functions
def _0_1_loss(seq1, seq2):
    loss = 0.0
    for i, j in zip(seq1, seq2):
        loss += 1 if i != j else 0
    return loss


def _predict(model, dataset, num_ner_tags):
    ret = []
    for i in range(len(dataset)):
        words, pos_tags, chunk_tags, ner_tags = dataset[i]
        _, best_seq = viterbi.viterbi_decode(
            words, pos_tags, chunk_tags, ner_tags, num_ner_tags, model, use_loss=False)
        ret += [best_seq]
    return ret


def _flatten(list2d):
    return list(itertools.chain.from_iterable(list2d))
