import numpy as np
import pickle as pkl
import os
import time


class VocDataset:
    """Store a set of images of Pascal VOC 2007 for image localization. Supports ``__len__`` and ``__getitem__``.

        Each image is assumed to contain only one object of the class of interest (it may contain other classes).

        Apart from the images and labels of Pascal VOC 2007, this class also requires a set of candidate bounding
        boxes to contain this class (e.g., from the output of selective search) and the IoU (intersection over union)
        of each candidate bounding box with the true bounding box, as well as features of the
        bounding box.

        The candidate boxes and their feature representation are directly loaded from disk.

        :param obj_class: Name of object class.
        :param label_file: Path to the pickle file with labels and bounding boxes.
        :param feature_dir: Directory with features.
            Each file is stored as ``{image-ID}_{obj-class}.npy``, where image-ID is the six digit image ID number.
        :param data_type: ``'train'`` or ``'val'``
        :param num_bboxes_per_image: Number of bounding boxes loaded per image. Default 1000.
        :param dim: Dimensionality of features. Default is 2304.
        """

    def __init__(self, obj_class, label_file, feature_dir, data_type,
                 num_bboxes_per_image=1000, dim=2304):
        self.dim = dim
        t1 = time.time()
        with open(label_file, 'rb') as f:
            temp = pkl.load(f)
        t2 = time.time()
        self.img_ids, self.labels, self.bboxes, self.projected_bboxes, self.ious = temp  # format of label file
        self.labels = np.asarray(self.labels, dtype=np.float32)
        self.bboxes = [np.asarray(bbs[:num_bboxes_per_image])
                       for bbs in self.bboxes]
        self.projected_bboxes = [np.asarray(bbs[:num_bboxes_per_image])
                                 for bbs in
                                 self.projected_bboxes]
        self.ious = [np.asarray(ious[:num_bboxes_per_image], dtype=np.float32)
                     for ious in self.ious]
        self.best_bbox_idxs = [ious.argmax() for ious in self.ious]

        # Shuffle and split
        idxs = _random_order(len(self.img_ids))
        split_point = len(idxs) * 3 // 4
        if data_type in ['train', 'train2014']:
            min_idx = 0
            max_idx = split_point
        else:
            min_idx = split_point
            max_idx = len(self.img_ids)
        idxs = idxs[min_idx:max_idx]
        self.img_ids = [self.img_ids[idx] for idx in idxs]
        self.labels = np.asarray([self.labels[idx] for idx in idxs], dtype=np.float32)
        self.bboxes = [self.bboxes[idx] for idx in idxs]
        self.projected_bboxes = [self.projected_bboxes[idx] for idx in idxs]
        self.ious = [self.ious[idx] for idx in idxs]
        self.best_bbox_idxs = [self.best_bbox_idxs[idx] for idx in idxs]

        self.features = [np.load(_voc_features_path(img_id, feature_dir, obj_class, extension='npy')).reshape((-1, dim))
                         for img_id in self.img_ids]
        t3 = time.time()

        self.images = [_VocImage(_feats, _ious) for (_feats, _ious) in zip(self.features, self.ious)]

        print('Loaded dataset: labels {:1.2f}s ; images {:1.2f}s.'.format(t2-t1, t3-t2))

        # all labels
        self.label_array = np.zeros((len(self.img_ids), num_bboxes_per_image),
                                    dtype=np.int)
        for i in range(len(self.images)):
            img = self.images[i]
            n = img.ious.shape[0]
            self.label_array[i, :n] = (img.ious > 0.5)

        # buffer for later use:
        self.outs_array_1 = np.zeros((len(self.img_ids), num_bboxes_per_image))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        return self.images[idx]


# Utilities
class _VocImage:
    """Image class, to represent an image."""
    def __init__(self, feats, ious):
        self.feats = feats
        self.ious = ious
        self.label_idx = ious.argmax()  # bbox with highest iou is considered as ground truth
        self.label_feats = feats[self.label_idx, :]


def _random_order(n):
    rng = np.random.RandomState(1)
    x = np.asarray(list(range(n)))
    rng.shuffle(x)
    return x


def _voc_features_path(file_num, feature_dir, obj_class, extension='npy'):
    # Pascal VOC stores each image using a 6 digit ID, zero padded if necessary
    return os.path.join(feature_dir, '{:06d}_{:s}.{:s}'.format(int(file_num), obj_class, extension))
