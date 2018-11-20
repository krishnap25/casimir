"""
Pure python implementation of Viterbi and top-K Viterbi algorithms found in viterbi.pyx.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import casimir.data.named_entity_recognition.features as features


def viterbi_decode(words, pos_tags, chunk_tags, ner_tags, num_tags, model, use_loss=True):
    """ Viterbi decoding to find maximum scoring label sequence

    :param words: sequence of words, as a part of input
    :param pos_tags: sequence of part of speech tags, as a part of input
    :param chunk_tags: sequence of chunk tags, as a part of input
    :param ner_tags: output sequence of NER tags
    :param num_tags: total number of possible NER tags
    :param model: Ndarray of weights
    :param use_loss: use loss-augmented inference with Hamming loss if true
    :return: highest score, best scoring label sequence
    """
    start_tag = -1
    num_pos = len(words)
    table = np.zeros((num_tags, num_pos))
    table.fill(np.nan)
    bp = np.zeros((num_tags, num_pos), dtype=np.int)
    bp.fill(-10000)
    # initialize
    for tag in range(num_tags):
        loss = 1 if (use_loss and tag != ner_tags[0]) else 0
        table[tag, 0] = loss + features.model_dot_features_at_position(
            model, words, 0, pos_tags, chunk_tags, start_tag, tag)

    # forward pass
    for position in range(1, num_pos):
        for tag in range(num_tags):
            loss = 1 if (use_loss and tag != ner_tags[position]) else 0
            largest = -float('inf')
            argmax = 0
            for prev_tag in range(num_tags):
                val = table[prev_tag, position-1] + loss + \
                      features.model_dot_features_at_position(
                          model, words, position, pos_tags, chunk_tags, prev_tag, tag)
                if val > largest:
                    largest = val
                    argmax = prev_tag
            table[tag, position] = largest
            bp[tag, position] = argmax

    # max
    largest = table[:, -1].max()
    argmax = table[:, -1].argmax()

    # reconstruct best sequence
    best_seq = [argmax]
    for position in range(num_pos - 1, 0, -1):
        argmax = bp[argmax, position]
        best_seq.insert(0, argmax)

    return largest, best_seq


def _find_k_best(k, arr):
    """Return k largest entries and indices of k largest entries of 2D-array arr."""
    flat_indices = np.argpartition(arr.ravel(), -k)[-k:]
    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
    max_elements = arr[row_indices, col_indices]
    max_elements_order = np.argsort(max_elements)[::-1]
    row_indices, col_indices = row_indices[max_elements_order], col_indices[max_elements_order]
    return arr[row_indices, col_indices], row_indices, col_indices


def viterbi_decode_top_k(num_highest_scores, words, pos_tags, chunk_tags, ner_tags, num_tags, model, use_loss=True):
    """Top-K Viterbi decoding to find K maximum scoring label sequences

    :param num_highest_scores: the parameter K in top-K inference
    :param words: sequence of words, as a part of input
    :param pos_tags: sequence of part of speech tags, as a part of input
    :param chunk_tags: sequence of chunk tags, as a part of input
    :param ner_tags: output sequence of NER tags
    :param num_tags: total number of possible NER tags
    :param model: Ndarray of weights
    :param use_loss: use loss-augmented inference with Hamming loss if true
    :return: highest score, best scoring label sequence
    """
    start_tag = -1
    num_pos = len(words)
    table = np.zeros((num_tags, num_pos, num_highest_scores))
    table.fill(np.nan)
    bp_tag = np.zeros((num_tags, num_pos, num_highest_scores), dtype=np.intp)
    bp_k = np.zeros((num_tags, num_pos, num_highest_scores), dtype=np.intp)
    scores = np.zeros((num_tags, num_highest_scores))  # to store top-K scores efficiently
    bp_tag.fill(-10000)
    bp_k.fill(-10000)
    # init
    for tag in range(num_tags):
        loss = 1 if (use_loss and tag != ner_tags[0]) else 0
        table[tag, 0, 0] = (
                loss + features.model_dot_features_at_position(model, words, 0, pos_tags, chunk_tags, start_tag, tag)
        )
    table[:, 0, 1:] = -np.inf

    # forward pass
    for position in range(1, num_pos):
        table[:, position, position + 1:num_highest_scores] = -np.inf
        for tag in range(num_tags):
            loss = 1 if (use_loss and tag != ner_tags[position]) else 0
            for prev_tag in range(num_tags):
                feat_dot_model = features.model_dot_features_at_position(model,
                                                                         words, position,
                                                                         pos_tags, chunk_tags,
                                                                         prev_tag, tag)
                scores[prev_tag, :] = loss + table[prev_tag, position - 1, :] + feat_dot_model
            # find k best from scores
            best_K, row_idxs, col_idxs = _find_k_best(num_highest_scores, scores)
            table[tag, position, :] = best_K
            bp_tag[tag, position, :] = row_idxs
            bp_k[tag, position, :] = col_idxs

    # max
    best_seqs = np.zeros((num_highest_scores, num_pos), dtype=np.intp)
    best_idxs = np.zeros((num_highest_scores, num_pos), dtype=np.intp)
    best_scores, row_idxs, col_idxs = _find_k_best(num_highest_scores, table[:, -1, :])
    best_seqs[:, -1] = row_idxs
    best_idxs[:, -1] = col_idxs

    # follow back-pointers
    for position in range(num_pos - 2, -1, -1):  # num_pos-2 to 0, both inclusive
        best_seqs[:, position] = bp_tag[best_seqs[:, position + 1],
                                        position + 1, best_idxs[:, position + 1]]
        best_idxs[:, position] = bp_k[best_seqs[:, position + 1],
                                      position + 1, best_idxs[:, position + 1]]
    return best_scores, best_seqs


def determine_weights_l2(smoothing_coefficient, scores):
    """Determine weights for l2 smoothing by projecting ``scores``/``smoothing_coefficient`` onto simplex.

    :param smoothing_coefficient: smoothing coefficient :math:`\mu`
    :param scores: Ndarray representing a vector :math:`p`
    :return: Projection of :math:`p/\mu` onto simplex
    """
    mu = smoothing_coefficient
    k = scores.shape[0]
    cumulative_sums = np.cumsum(scores)
    rho = (1 - cumulative_sums[-1] / mu) / k
    for k in range(0, k - 1):
        if cumulative_sums[k] - (k + 1) * scores[k + 1] >= mu:
            rho = (1 - cumulative_sums[k] / mu) / (k + 1)
            break
    return np.maximum(0, scores / mu + rho)
