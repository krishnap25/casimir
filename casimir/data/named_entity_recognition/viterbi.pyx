import numpy as np
cimport numpy as np
from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair
from libc cimport math
from casimir.data.named_entity_recognition import features

def viterbi_decode(words, pos_tags, chunk_tags, ner_tags,
                   int num_tags, np.ndarray[np.float64_t, ndim=1] model, use_loss=True):
    """Viterbi decoding to find maximum scoring label sequence

    :param words: sequence of words, as a part of input
    :param pos_tags: sequence of part of speech tags, as a part of input
    :param chunk_tags: sequence of chunk tags, as a part of input
    :param ner_tags: output sequence of NER tags
    :param num_tags: total number of possible NER tags
    :param model: Ndarray of weights
    :param use_loss: use loss-augmented inference with Hamming loss if true
    :return: highest score, best scoring label sequence
    """
    # declarations
    cdef:
        np.ndarray[np.float64_t, ndim=2] table
        double[:, :] table_view
        np.ndarray[np.int_t, ndim=2] bp
        np.ndarray[np.int_t, ndim=1] best_seq
        long[:, :] bp_view
        long[:] best_seq_view
        int num_pos, start_tag, i, j, k, loss, argmax, prev_tag, tag, position
        double largest, val
    # init
    start_tag = -1
    num_pos = len(words)
    table = np.zeros((num_tags, num_pos))
    table.fill(np.nan)
    table_view = table
    bp = np.zeros((num_tags, num_pos), dtype=np.int)
    bp_view = bp
    bp_view[:, :] = -100000
    best_seq = np.empty((num_pos,), dtype=np.int)
    best_seq_view = best_seq

    # initialize
    for tag in range(num_tags):
        loss = 1 if (use_loss and tag != ner_tags[0]) else 0
        table_view[tag, 0] = loss + features.model_dot_features_at_position(
            model, words, 0, pos_tags, chunk_tags, start_tag, tag)

    # forward pass
    for position in range(1, num_pos):
        for tag in range(num_tags):
            loss = 1 if (use_loss and tag != ner_tags[position]) else 0
            largest = -math.INFINITY
            argmax = 0
            for prev_tag in range(num_tags):
                val = table_view[prev_tag, position-1] + loss + \
                      features.model_dot_features_at_position(
                        model, words, position, pos_tags, chunk_tags, prev_tag, tag)
                if val > largest:
                    largest = val
                    argmax = prev_tag
            table_view[tag, position] = largest
            bp_view[tag, position] = argmax

    # max
    largest = -math.INFINITY
    argmax = 0
    for tag in range(num_tags):
        if table[tag, num_pos-1] > largest:
            largest = table_view[tag, num_pos-1]
            argmax = tag

    # follow back-pointers
    best_seq_view[num_pos-1] = argmax
    for position in range(num_pos-1, 0, -1):
        argmax = bp_view[argmax, position]
        best_seq_view[position-1] = argmax

    return largest, best_seq


def viterbi_decode_top_k(int num_highest_scores, words, pos_tags, chunk_tags, ner_tags, num_tags,
                        np.ndarray[np.float64_t, ndim=1] model, use_loss=True):
    """Top-K Viterbi decoding to find :math:`K` maximum scoring label sequences

    :param num_highest_scores: the parameter :math:`K` in top-K inference
    :param words: sequence of words, as a part of input
    :param pos_tags: sequence of part of speech tags, as a part of input
    :param chunk_tags: sequence of chunk tags, as a part of input
    :param ner_tags: output sequence of NER tags
    :param num_tags: total number of possible NER tags
    :param model: ``numpy.ndarray`` of weights
    :param boolean use_loss: use loss-augmented inference with Hamming loss if ``True``
    :return: :math:`K` highest scores and the corresponding label sequences
    """
    cdef:
        np.ndarray[np.float64_t, ndim=3] table
        double[:, :, :] table_view
        np.ndarray[np.int_t, ndim=3] bp_tag, bp_k
        np.ndarray[np.int_t, ndim=2] best_seqs, best_idxs
        long[:, :, :] bp_tag_view, bp_k_view
        long[:, :] best_seqs_view, best_idxs_view
        int num_pos, start_tag, i, j, k, loss, argmax, prev_tag, tag, position
        double largest, val, feat_dot_model
        priority_queue[pair[double, pair[int, int]]] pq
        pair[int, int] idxs
        pair[double, pair[int, int]] pq_entry
        np.ndarray[np.float64_t, ndim=1] best_scores
        double[:] best_scores_view
    # initialize variables
    start_tag = -1
    num_pos = len(words)
    table = np.zeros((num_tags, num_pos, num_highest_scores))
    table.fill(np.nan)
    table_view = table
    bp_tag = np.zeros((num_tags, num_pos, num_highest_scores), dtype=np.int)
    bp_k = np.zeros((num_tags, num_pos, num_highest_scores), dtype=np.int)
    bp_tag_view = bp_tag
    bp_k_view = bp_k
    bp_tag_view[:, :, :] = -10000
    bp_k_view[:, :, :] = -10000
    best_scores = np.zeros(num_highest_scores)  # to store top-K scores
    best_scores_view = best_scores
    best_seqs = np.zeros((num_highest_scores, num_pos), dtype=np.int)
    best_seqs_view = best_seqs
    best_idxs = np.zeros((num_highest_scores, num_pos), dtype=np.int)
    best_idxs_view = best_idxs

    # initialize table for top-K Viterbi algorithm
    for tag in range(num_tags):
        loss = 1 if (use_loss and tag != ner_tags[0]) else 0
        table_view[tag, 0, 0] = (
                loss + features.model_dot_features_at_position(model, words, 0, pos_tags, chunk_tags, start_tag, tag)
        )
        for k in range(1, num_highest_scores):
            table_view[tag, 0, k] = -math.INFINITY

    # forward pass
    for position in range(1, num_pos):
        for tag in range(num_tags):
            for k in range(position+1, num_highest_scores):
                table_view[tag, position, k] = -math.INFINITY
        # loop
        for tag in range(num_tags):
            pq = priority_queue[pair[double, pair[int, int]]]()  # clear pq
            loss = 1 if (use_loss and tag != ner_tags[position]) else 0

            # loop over previous tag and k
            for prev_tag in range(num_tags):
                feat_dot_model = features.model_dot_features_at_position(model,
                                                                         words, position,
                                                                         pos_tags, chunk_tags,
                                                                         prev_tag, tag)
                for k in range(num_highest_scores):
                    val = loss + table_view[prev_tag, position-1, k] + feat_dot_model
                    pq.push(pair[double, pair[int, int]](
                        val, pair[int, int](prev_tag, k)))
            # find k best from scores
            for k in range(num_highest_scores):
                pq_entry = pq.top()
                idxs = pq_entry.second
                table_view[tag, position, k] = pq_entry.first
                bp_tag_view[tag, position, k] = idxs.first
                bp_k_view[tag, position, k] = idxs.second
                pq.pop()

    # compute max
    pq = priority_queue[pair[double, pair[int, int]]]()  # clear priority queue
    for tag in range(num_tags):
        for k in range(num_highest_scores):
            pq.push(pair[double, pair[int, int]](
                table_view[tag, num_pos-1, k], pair[int, int](tag, k)))
    # pop k best scores
    for k in range(num_highest_scores):
        pq_entry = pq.top()
        idxs = pq_entry.second
        best_scores_view[k] = pq_entry.first
        best_seqs_view[k, num_pos-1] = idxs.first
        best_idxs_view[k, num_pos-1] = idxs.second
        pq.pop()

    # follow back-pointers to compute sequences
    for position in range(num_pos-2, -1, -1):  # num_pos-2 to 0, both inclusive
        for k in range(num_highest_scores):
            best_seqs_view[k, position] = bp_tag_view[best_seqs_view[k, position+1],
                                                      position+1, best_idxs_view[k, position+1]]
            best_idxs_view[k, position] = bp_k_view[best_seqs_view[k, position + 1],
                                                    position+1, best_idxs_view[k, position+1]]

    return best_scores, best_seqs


def determine_weights_l2(double smoothing_coefficient, np.ndarray[np.float64_t, ndim=1] scores):
    """Determine weights for l2 smoothing by projecting ``scores``/``smoothing_coefficient`` onto simplex.

    :param smoothing_coefficient: smoothing coefficient :math:`\mu`
    :param scores: Ndarray representing a vector :math:`p`
    :return: Projection of :math:`p/\mu` onto simplex
    """
    cdef double mu = smoothing_coefficient
    cdef int num_highest_scores = scores.shape[0]
    cumulative_sums = np.cumsum(scores)
    cdef double rho = ( 1- cumulative_sums[num_highest_scores-1] / mu) / num_highest_scores
    cdef int k
    for k in range(0, num_highest_scores - 1):
        if cumulative_sums[k] - (k + 1) * scores[k+1] >= mu:
            rho = (1 - cumulative_sums[k] / mu) / (k + 1)
            break
    return np.maximum(0, scores / mu + rho)
