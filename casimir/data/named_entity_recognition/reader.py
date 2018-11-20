from __future__ import absolute_import, division, print_function
import numpy as np


def generate_counts_and_feature_map(lines, threshold=1):
    """Pre-processing step to read data in CoNLL-2003 format."""
    counts = {}
    word_to_idx, idx_to_word = {}, {}
    pos_tags, chunk_tags, ner_tags = {}, {}, {}
    num_sentences = 0
    doc_start = False
    for line in lines:
        if line[0:10] == u'-DOCSTART-':
            doc_start = True
            continue
        elif doc_start:
            doc_start = False
            continue
        elif len(line) == 1 or line == u'\n':
            num_sentences += 1
        elif not (line.isspace() or (len(line) > 10 and line[0:10] == u'-DOCSTART-') or len(line) == 0):
            line = line.rstrip('\n').split()
            word = line[0]
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1
            if line[1] not in pos_tags:
                pos_tags[line[1]] = len(pos_tags)
            if line[2] not in chunk_tags:
                chunk_tags[line[2]] = len(chunk_tags)
            if line[3] not in ner_tags:
                ner_tags[line[3]] = len(ner_tags)
    num_ner_tags = len(ner_tags)
    idx = 1
    word_to_idx['<unk>'] = 0
    idx_to_word[0] = '<unk>'
    word_to_idx['START'], word_to_idx['STOP'] = -1, -2
    idx_to_word[-1], idx_to_word[-2] = 'START', 'STOP'
    pos_tags['START'], chunk_tags['START'], ner_tags['START'] = -1, -1, -1
    pos_tags['STOP'], chunk_tags['STOP'], ner_tags['STOP'] = -2, -2, -2
    for word, count in counts.items():
        if count > threshold:
            word_to_idx[word] = idx
            idx_to_word[idx] = word
            idx += 1
    idx_to_ner_tags = {i: tag for tag, i in ner_tags.items()}
    return (word_to_idx, idx_to_word, counts, num_sentences,
            pos_tags, chunk_tags, ner_tags, idx_to_ner_tags, num_ner_tags)


def _find_num_sentences(lines):
    i = 0  # num sentences
    j = 0  # num doc_start
    doc_start = False
    for line in lines:
        if line[0:10] == u'-DOCSTART-':
            j += 1
            doc_start = True
            continue
        elif doc_start:
            doc_start = False
            continue
        elif line.isspace() or (len(line) > 10 and line[0:10] == u'-DOCSTART-'):
            i += 1
    return i


class NerDataset:
    """Class to parse CoNLL-2003 data and allow random access into dataset.

        Requires output of ``generate_counts_and_feature_map``.
    """
    def __init__(self, lines, word_to_idx, pos_to_idx, chunk_to_idx, ner_to_idx):
        # words, posTags, chunkTags, nerTags are all np.arrays of lists.
        # each list represents a sentence
        num_sentences = _find_num_sentences(lines)
        self.words = np.empty((num_sentences,), dtype=np.object)
        self.pos_tags = np.empty((num_sentences,), dtype=np.object)
        self.chunk_tags = np.empty((num_sentences,), dtype=np.object)
        self.ner_tags = np.empty((num_sentences,), dtype=np.object)
        words, pos, chunk, ner = [], [], [], []
        i = 0
        doc_start = False
        for line in lines:
            if line[0:10] == u'-DOCSTART-':
                doc_start = True
                continue
            elif doc_start:
                doc_start = False
                continue
            elif not (line.isspace() or (len(line) > 10 and line[0:10] == u'-DOCSTART-')):
                line = line.rstrip('\n').split()
                word = line[0]
                if word in word_to_idx:
                    words += [word_to_idx[word]]
                else:
                    words += [word_to_idx['<unk>']]
                pos += [pos_to_idx[line[1]]]
                chunk += [chunk_to_idx[line[2]]]
                ner += [ner_to_idx[line[3]]]
            else:
                self.words[i] = words  # + [stop]
                self.pos_tags[i] = pos  # + [stop_pos]
                self.chunk_tags[i] = chunk  # + [stop_chunk]
                self.ner_tags[i] = ner  # + [stop_ner]
                words, pos, chunk, ner = [], [], [], []
                i += 1

    def __getitem__(self, idx):
        return self.words[idx], self.pos_tags[idx], self.chunk_tags[idx], self.ner_tags[idx]

    def __len__(self):
        return self.words.shape[0]
