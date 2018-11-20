from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.metrics import average_precision_score
from . import reader
import casimir.optim.incremental_first_order_oracle as ifo


def create_loc_ifo_from_data(obj_class, label_file, features_dir, dimension=2304,
                             smoothing_coefficient=None, num_highest_scores=5):
    """Create a smoothed incremental first order oracle for train and validation sets for Pascal VOC 2007
    given candidate bounding boxes and features.

    :param obj_class: Name of object class.
    :param label_file: Pickled file with candidate bounding boxes and their IoUs with ground truth.
    :param features_dir: Directory with feature representation of candidate bounding boxes.
            Each file is stored as ``{image-ID}_{obj-class}.npy``, where image-ID is the six digit image ID number.
    :param dimension: Dimension of feature representation. Default is 2304, a hard-coded value.
    :param smoothing_coefficient: Smoothing coefficient to initialize Smoothed IFOs. Default ``None``.
    :param num_highest_scores: The parameter :math:`K` used by the top-K inference  algorithm. Default 5.
    :return: Train, dev and test Smoothed IFOs. Dev (test) IFO is ``None`` if dev (test) file is ``None``.
    """
    train_set = reader.VocDataset(obj_class, label_file, features_dir, 'train', dim=dimension)
    dev_set = reader.VocDataset(obj_class, label_file, features_dir, 'val', dim=dimension)

    # Create IFOs. Only train IFO requires smoothing information
    train_ifo = LocalizationIfo(train_set, smoothing_coefficient, num_highest_scores)
    dev_ifo = LocalizationIfo(dev_set)
    test_ifo = None  # test set labels are not available
    return train_ifo, dev_ifo, test_ifo


class LocalizationIfo(ifo.SmoothedIncrementalFirstOrderOracle):
    """Create the smoothed IFO object for the structural hinge loss object localization with Pascal VOC.

    Supports both non-smooth and :math:`\ell_2` smoothed versions of the structural hinge loss
    but not entropy smoothing.

    :param voc_dataset: ``reader.VocDataset`` object that supports ``__len__`` and ``__getitem__`` methods.
    :param smoothing_coefficient: The amount of smoothing, :math:`\mu`.
    :param num_highest_scores: The value :math:`K` for top-:math:`K` inference.
    """
    def __init__(self,
                 voc_dataset,
                 smoothing_coefficient=None,
                 num_highest_scores=10):
        super(LocalizationIfo, self).__init__(smoothing_coefficient)
        self.voc_dataset = voc_dataset
        self.num_highest_scores = num_highest_scores

    def __len__(self):
        return len(self.voc_dataset)

    def function_value(self, model, idx):
        image = self.voc_dataset[idx]
        scores = np.matmul(image.feats, model) + (1 - image.ious)
        gold_score = scores[image.label_idx]

        if self.smoothing_coefficient is not None:
            # l2 smoothing
            top_scores, top_idxs = _top_k(self.num_highest_scores, scores)
            weights = determine_weights_l2(self.smoothing_coefficient, top_scores)
            fn_val = np.asarray([top_scores[0] - gold_score, np.dot(weights, top_scores) - gold_score])

        else:
            # no smoothing
            fn_val = scores.max() - gold_score

        return fn_val

    def gradient(self, model, idx):
        image = self.voc_dataset[idx]
        scores = np.matmul(image.feats, model) + (1 - image.ious)

        if self.smoothing_coefficient is not None:
            # l2 smoothing
            top_scores, top_idxs = _top_k(self.num_highest_scores, scores)
            weights = determine_weights_l2(self.smoothing_coefficient, top_scores)
            feats = image.feats[top_idxs, :]
            return np.matmul(feats.T, weights) - image.label_feats

        else:
            # no smoothing
            idx = scores.argmax()
            grad = image.feats[idx, :] - image.label_feats

        return grad

    def linear_minimization_oracle(self, model, idx):
        image = self.voc_dataset[idx]
        scores = np.matmul(image.feats, model) + (1 - image.ious)
        idx = scores.argmax()
        return image.label_feats - image.feats[idx, :], 1 - image.ious[idx]

    def evaluation_function(self, model):
        """Return average IoU, localization accuracy and average precision."""
        correct = 0.0
        ret = 0.0
        dataset = self.voc_dataset
        for i in range(len(dataset)):
            image = dataset[i]
            scores = np.matmul(image.feats, model)
            idx = scores.argmax()
            ret += image.ious[idx]
            correct += 1 if image.ious[idx] >= 0.5 else 0

            n = len(scores)
            dataset.outs_array_1[i, :n] = scores  # raw scores to compute AP

        return (ret / len(dataset), correct / len(dataset),
                average_precision_score(dataset.label_array, dataset.outs_array_1, average='micro'))


# Utility functions
def _top_k(k, outs):
    """Return the top k items in input Ndarray ``outs`` and the corresponding indices."""
    max_idxs_unsorted = np.argpartition(outs, -k)[-k:]  # unsorted
    max_idxs = max_idxs_unsorted[np.argsort(
        outs[max_idxs_unsorted])[::-1]]  # sorted
    top_k = outs[max_idxs]
    return top_k, max_idxs


def determine_weights_l2(smoothing_coefficient, scores):
    mu = smoothing_coefficient
    k = scores.shape[0]
    cumulative_sums = np.cumsum(scores)
    rho = (1 - cumulative_sums[-1] / mu) / k
    for k in range(0, k - 1):
        if cumulative_sums[k] - (k + 1) * scores[k + 1] >= mu:
            rho = (1 - cumulative_sums[k] / mu) / (k + 1)
            break
    return np.maximum(0, scores / mu + rho)
