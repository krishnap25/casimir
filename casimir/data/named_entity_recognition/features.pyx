import numpy as np
cimport numpy as np


cpdef int hash_function(int a1, int a2, int a3, int dim):
    """Return hash of triple ``(a1, a2, a3)`` of integers to size ``dim``"""
    cdef int hashcode = 17
    hashcode = 31 * hashcode + a1
    hashcode = 31 * hashcode + a2
    hashcode = 31 * hashcode + a3
    return hashcode % dim

  
cpdef _generate_features(words, int position, pos_tags, chunk_tags, int previous_ner_tag, int current_ner_tag, int dim):
    """Return sparse feature vector of indices (padded by -1) corresponding to value 1"""
    cdef:
        int length = len(words)
        int current_word = words[position], current_pos = pos_tags[position], current_chunk = chunk_tags[position]
        int prev_word, next_word, prev_pos, next_pos, prev_chunk, next_chunk
        int i
        # return
        int return_size = 100 # conservative size of maximum number of features at a given position
        np.ndarray to_return = np.empty(return_size, dtype=np.int32)
        int[:] ret = to_return
    if position == 0:
        prev_word = -1
        prev_pos = -1
        prev_chunk = -1
    else:
        prev_word = words[position-1]
        prev_pos = pos_tags[position-1]
        prev_chunk = chunk_tags[position-1]
    if position == length - 1:
        next_word = -2
        next_pos = -2
        next_chunk = -2
    else:
        next_word = words[position+1]
        next_pos = pos_tags[position+1]
        next_chunk = chunk_tags[position+1]

    # initialize to -1
    ret[:] = -1
    i = 0
    # word - tag 
    ret[i] = hash_function(i, current_word, current_ner_tag, dim)
    i += 1
    ret[i] = hash_function(i, current_word, previous_ner_tag, dim)
    i += 1
    if position > 1:
        ret[i] = hash_function(i, words[position-2], current_ner_tag, dim)
        i += 1
        ret[i] = hash_function(i, words[position-2], previous_ner_tag, dim)
        i += 1
    else:
        ret[i] = 0
        i += 1
        ret[i] = 0
        i += 1
    if position < length-2:
        ret[i] = hash_function(i, words[position+2], current_ner_tag, dim)
        i += 1
        ret[i] = hash_function(i, words[position+2], previous_ner_tag, dim)
        i += 1
    else:
        ret[i] = 0
        i += 1
        ret[i] = 0
        i += 1
    # previous word - tag
    ret[i] = hash_function(i, prev_word, current_ner_tag, dim)
    i += 1
    ret[i] = hash_function(i, prev_word, previous_ner_tag, dim)
    i += 1
    # next word - tag
    ret[i] = hash_function(i, next_word, current_ner_tag, dim)
    i += 1
    ret[i] = hash_function(i, next_word, previous_ner_tag, dim)
    i += 1
    # named_entity_recognition-named_entity_recognition
    ret[i] = hash_function(i, previous_ner_tag, current_ner_tag, dim)
    i += 1
    # pos-pos and chunk-chunk
    ret[i] = hash_function(i, prev_pos, current_pos, dim)
    i += 1
    ret[i] = hash_function(i, current_pos, next_pos, dim)
    i += 1
    ret[i] = hash_function(i, prev_chunk, current_chunk, dim)
    i += 1
    ret[i] = hash_function(i, current_chunk, next_chunk, dim)
    i += 1
    # named_entity_recognition-pos and named_entity_recognition-chunk
    ret[i] = hash_function(i, current_ner_tag, current_pos, dim)
    i += 1
    ret[i] = hash_function(i, current_ner_tag, next_pos, dim)
    i += 1
    ret[i] = hash_function(i, current_ner_tag, prev_pos, dim)
    i += 1
    ret[i] = hash_function(i, current_ner_tag, current_chunk, dim)
    i += 1
    ret[i] = hash_function(i, current_ner_tag, next_chunk, dim)
    i += 1
    ret[i] = hash_function(i, current_ner_tag, prev_chunk, dim)
    i += 1
    # prev named_entity_recognition-pos and prev named_entity_recognition-chunk
    ret[i] = hash_function(i, previous_ner_tag, current_pos, dim)
    i += 1
    ret[i] = hash_function(i, previous_ner_tag, next_pos, dim)
    i += 1
    ret[i] = hash_function(i, previous_ner_tag, prev_pos, dim)
    i += 1
    ret[i] = hash_function(i, previous_ner_tag, current_chunk, dim)
    i += 1
    ret[i] = hash_function(i, previous_ner_tag, next_chunk, dim)
    i += 1
    ret[i] = hash_function(i, previous_ner_tag, prev_chunk, dim)
    i += 1

    return to_return


cpdef double model_dot_features_at_position(np.ndarray[np.float64_t, ndim=1] model, words, int position,
                                            pos_tags, chunk_tags, int previous_ner_tag, int current_ner_tag):
    """Compute dot product between ``model`` and feature vector of (words, pos_tags, chunk_tags) at ``position``.
    
    Letting :math:`w` represent the model, :math:`x` the sentence (words, pos_tags, chunk_tags) and 
    :math:`y` the labels (ner_tags). The feature vector :math:`\Phi` is then, 
    :math:`\Phi(x, y) = sum_i \Phi_i(x, y_{i-1}, y_i`. 
    This function computes :math:`\lange w, \Phi_i(x, y_{i-1}, y_i) \rangle` where :math:`i` is the input 
    ``position``.
    The start of the sequence is assigned the special value -1 (start_tag).

    
    :param model: Ndarrray of weights of linear model :math:`w`
    :param words: sequence of words, as a part of input :math:`x`
    :param position: position :math:`i` at which to compute feature vector
    :param pos_tags: part of speech tags, as a part of input :math:`x`
    :param chunk_tags: chunk tags, as a part of input :math:`x`
    :param previous_ner_tag: :math:`y_{i-1}`
    :param current_ner_tag: :math:`y_{i}`
    :return: dot product between model and feature vector at given position
    """
    cdef:
        int dim = model.shape[0]
        int[:] features
        double[:] w = model
        int i
        double dot = 0.0
    features = _generate_features(words, position, pos_tags, chunk_tags,
                                 previous_ner_tag, current_ner_tag, dim)
    for i in range(len(features)):
        if features[i] < 0:
            break
        dot += w[features[i]]
    return dot

cpdef np.ndarray[np.float64_t, ndim=1] generate_feature_vector(words, pos_tags, chunk_tags, ner_tags, int dim):
    """Compute feature vector of (words, pos_tags, chunk_tags) and ner_tags of given dimensionality.
    
    Letting :math:`x` represent the sentence (words, pos_tags, chunk_tags) and 
    :math:`y` the labels (ner_tags). 
    This function computes the feature vector :math:`\Phi` given by 
    :math:`\Phi(x, y) = sum_i \Phi_i(x, y_{i-1}, y_i`. 

    
    :param words: sequence of words, as a part of input :math:`x`
    :param pos_tags: part of speech tags, as a part of input :math:`x`
    :param chunk_tags: chunk tags, as a part of input :math:`x`
    :param ner_tags: the label sequence, :math:`y`
    :param dim: dimensionality of feature vector
    :return: feature vector of the given sentence and label sequence 
    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] feature_vector = np.empty(dim, dtype=np.float64)
        double[:] feature_vector_view = feature_vector
        int[:] features
        int length = len(words), i, j
    feature_vector_view[:] = 0
    for i in range(length):
        if i != 0:
            features = _generate_features(words, i, pos_tags, chunk_tags,
                                           ner_tags[i - 1], ner_tags[i], dim)
        else:
            features = _generate_features(words, i, pos_tags, chunk_tags,
                                           -1, ner_tags[i], dim)
        for j in range(len(features)):
            if features[j] < 0:
                break
            feature_vector_view[features[j]] += 1
    return feature_vector

cpdef double model_dot_features(np.ndarray[np.float64_t, ndim=1] model, words, pos_tags, chunk_tags, ner_tags):
    """Compute dot product between ``model`` and feature vector of (``words``, ``pos_tags``, ``chunk_tags``) and ``ner_tags``.
    
    Letting :math:`w` represent the model, :math:`x` the sentence (words, pos_tags, chunk_tags) and 
    :math:`y` the labels (ner_tags). The feature vector :math:`\Phi` is then, 
    :math:`\Phi(x, y) = sum_i \Phi_i(x, y_{i-1}, y_i`. 
    This function computes :math:`\lange w, \Phi_i(x, y_{i-1}, y_i) \rangle` where :math:`i` is the input 
    ``position``.

    
    :param model: Ndarrray of weights of linear model :math:`w`
    :param words: sequence of words, as a part of input :math:`x`
    :param pos_tags: part of speech tags, as a part of input :math:`x`
    :param chunk_tags: chunk tags, as a part of input :math:`x`
    :param ner_tags: label sequence :math:`y`
    :return: dot product between model :math:`y` and feature vector :math:`\Phi(x, y)` 
    """
    cdef:
        int dim = model.shape[0]
        int position, i, sentence_len, previous_ner_tag, current_ner_tag
        double dot = 0.0
        int[:] features
        double[:] w = model
    sentence_len = len(words)
    previous_ner_tag = -1
    for position in range(sentence_len):
        current_ner_tag = ner_tags[position]
        features = _generate_features(words, position, pos_tags, chunk_tags,
                                      previous_ner_tag, current_ner_tag, dim)
        for i in range(len(features)):
            if features[i] < 0:
                break
            dot += w[features[i]]
        previous_ner_tag = current_ner_tag
    return dot
