"""
Load and preprocess IMDB data.

Based on CEN implementation
"""

import os
import json
from collections import Counter

import numpy as np
from torch.utils.data import Dataset
from utils import get_freq_itemsets, build_satisfiability_matrix
from .preprocess import pad_sequences

class IMDB(Dataset):
    # def __init__(self, x, c, y, S):
    def __init__(self, data):
        x = data['x']
        c = data['c']
        y = data['y']
        S = data['S']

        self.S = S.astype('f4')
        self.x = x
        self.y = y
        self.context = c

        self.S = self.S / self.S.sum(0, keepdims=True)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # return the corresponding row of S, context, x, and label.

        return {
            'S': self.S[idx],
            'context': self.context[idx],
            'x': self.x[idx],
            'y': self.y[idx]
        }

def load_imdb_init(args, nb_words=None, skip_top=0,
            start_char=1, oov_char=2, index_from=3):
    """
    Loads IMDB data, adds start characters/reindexes, and
    filters OOV words
    """
    path = args['raw_file']
    f = np.load(path)

    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']

    # re-index
    if start_char is not None:
        x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
        x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
    elif index_from:
        x_train = [[w + index_from for w in x] for x in x_train]
        x_test = [[w + index_from for w in x] for x in x_test]

    # filter OOV words
    if not nb_words:
        nb_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test]))

    print(nb_words)

    if oov_char is not None:
        x_train = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x]
                    for x in x_train]
        x_test = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x]
                    for x in x_test]
    else:
        raise NotImplementedError

        # keras just removes the words...

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def get_interpretable_features(sequences, vocab, binary=True, counts=True,
                            index_from=3):
    """
    Get interpretable features from sequences, in terms of word presence/word counts

    Arguments:
    - sequences: list of word indices
    - vocab: mapping of words to index
    - binary: whether to include binary feature (<word> in sequence)
    - counts: whether to include count features (<word> appears <x> times in sequence)
    """

    # reverse
    word_mapping = {v + (index_from - 1): w for w, v in vocab.items()}

    features = []
    for seq in sequences:
        feats = []
        c = Counter(seq)

        if binary:
            binary_feats = [
                f"'{word_mapping[i]}' present"
                for i in c.keys() if i in word_mapping
            ]
            feats.extend(binary_feats)

        if counts:
            # count_feats = [
            #     f"'{word_mapping[i]}' appears {v} times"
            #     for i, v in c.items() if (i in word_mapping
            # ]
            count_feats = [
                f"'{word_mapping[i]}' appears >= 3 times"
                for i, v in c.items() if (i in word_mapping and v >= 3)
            ]
            feats.extend(count_feats)

        # TODO: non-binary

        features.append(feats)

    return features

def load_imdb(args):
    (x_train, y_train), (x_test, y_test) = load_imdb_init(args,
                                                          nb_words=args['max_vocab'],
                                                          skip_top=0,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)

    vocab = json.load(open(args['vocab_file'], 'r'))

    if args['max_vocab'] is not None:
        vocab_size = args['max_vocab'] + 3
    else:
        vocab_size = len(vocab) + 3

    # get interpretable features and get itemsets
    train_features = get_interpretable_features(x_train, vocab)
    test_features = get_interpretable_features(x_test, vocab)

    # truncate
    x_train = pad_sequences(x_train, maxlen=50, truncating='post', dtype=np.int64)
    x_test = pad_sequences(x_test, maxlen=50, truncating='post', dtype=np.int64)

    # split train into train and validation
    N_TRAIN = 22500
    N_VALID = 2500

    seed = args['seed']
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(x_train))

    X = [train_features[i] for i in order]
    Y = y_train[order]
    C = x_train[order]

    x_train = X[:N_TRAIN]
    y_train = Y[:N_TRAIN]
    c_train = C[:N_TRAIN]

    x_valid = X[N_TRAIN:]
    y_valid = Y[N_TRAIN:]
    c_valid = C[N_TRAIN:]

    c_test = x_test
    x_test = test_features

    # y_train is a list, change to list of 2
    y_train = np.array([
        [0, 1] if y == 1 else [1, 0]
        for y in y_train
    ])
    y_valid = np.array([
        [0, 1] if y == 1 else [1, 0]
        for y in y_valid
    ])
    y_test = np.array([
        [0, 1] if y == 1 else [1, 0]
        for y in y_test
    ])

    # get antecedents from train data
    min_support = args['min_support']
    max_lhs = args['max_lhs']
    antes = get_freq_itemsets(x_train, y_train, min_support=min_support, max_lhs=max_lhs)

    # build satisfiability matrix
    S_train = build_satisfiability_matrix(x_train, antes)
    S_valid = build_satisfiability_matrix(x_valid, antes)
    S_test = build_satisfiability_matrix(x_test, antes)

    train_data = {
        'x': x_train,
        'y': y_train,
        'c': c_train,
        'S': S_train,
    }
    valid_data = {
        'x': x_valid,
        'y': y_valid,
        'c': c_valid,
        'S': S_valid,
    }
    test_data = {
        'x': x_test,
        'y': y_test,
        'c': c_test,
        'S': S_test,
    }

    return {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'antes': antes,
        'vocab_size': vocab_size,
        'order': order
    }
