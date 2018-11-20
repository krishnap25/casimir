Reproducing experiments in the paper
====================================
The files ``main_ner.py`` and ``main_loc.py`` can be used to run experiments on the tasks of
named entity recognition (abbreviated NER) and visual object localization (abbreviated LOC) respectively.
The command line options are described `on this page <api_detailed/struct_pred.html>`_.
The scripts in the folder ``scripts/`` can be used to reproduce the experiments in the paper.

Data to reproduce experiments
-----------------------------

Named Entity Recognition

    We use the `CoNLL 2003 dataset <http://www.aclweb.org/anthology/W03-0419.pdf>`_.
    Get the data `here <https://www.clips.uantwerpen.be/conll2003/ner/>`_
    and follow instructions in the readme to get the data.
    Since the data is copyrighted, we cannot provide the extracted data here.

Visual Object Localization

    We use `Pascal VOC 2007 dataset <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/>`_.
    The pre-processed data and extracted features can be found `at this link <https://drive.google.com/drive/folders/1SJkzaCdqmNyGjz4BQniuZASP9Oi41GC2?usp=sharing>`_ on Google drive.
    Detailed instructions are given below.

Instructions to use pre-processed data for localization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data used for the visual object localization experiments is
`hosted here <https://drive.google.com/drive/folders/1SJkzaCdqmNyGjz4BQniuZASP9Oi41GC2?usp=sharing>`_.
Here are details about the contents of the shared data:

- ``bboxes/`` or ``bboxes.tgz``:
    This folder (size 341 MB uncompressed or 113 MB compressed) contains information about 1000
    candidate bounding boxes output by selective search. The information incldues the position of each bounding box and its
    IoU (intersection over union) with the ground truth bounding box.
    This folder is to be given as the argument ``--bbox-dir`` to ``main_loc.py``.
- ``features/``:
    This folder (size 37 GB) contains extracted features for each bounding box for each image over all classes of Pascal VOC.
    Full details of the extracted features can be found in the long version of the paper.   This folder is to be given as the argument ``--features-dir`` to ``main_loc.py``.
- ``sliding_bboxes/``, ``sliding_bboxes.tgz``, ``sliding_features/``:
    The counterparts to the above where the output of selective search
    is samples from a sliding window. The stride of the sliding window is chosen so that we get around 1300 candidate bounding boxes.
    This data can be used as a drop in replacement for ``bboxes/`` and ``features/`` respectively.
- Notes:
    - To run experiments for a certain class, one only needs the features of images for that class. The names of each individual file in the features directory is of the form ``{image-ID}_{obj-class}.npy``.
    - The files were uploaded with `Gdrive (command line interface for Google Drive) <https://github.com/prasmussen/gdrive>`_.




