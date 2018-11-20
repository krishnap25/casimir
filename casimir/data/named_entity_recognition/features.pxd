cimport numpy as np
import numpy as np

cpdef double model_dot_features_at_position(np.ndarray[np.float64_t, ndim=1] w, words, int position,
                                            pos_tags, chunk_tags, int previous_ner_tag, int current_ner_tag)

cpdef np.ndarray[np.float64_t, ndim=1] generate_feature_vector(words, pos_tags, chunk_tags, ner_tags, int dim)

cpdef double model_dot_features(np.ndarray[np.float64_t, ndim=1] w, words, pos_tags, chunk_tags, ner_tags)

