import csv
import os
import sys

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from utils import get_freq_itemsets, build_satisfiability_matrix
from support2_mod import load_data as load_sup

support2_keys = [
    'age',
    'temp',
    'crea',
    'sod',
    'hrt',
    'resp',
    'meanbp',
    'wblc',
    'sps',
    'surv2m',
    'surv6m',
    'ph',
    'pafi',
    'alb',
    'bili',
]

def normalize(train, valid, test):
    """
    Normalize feature values to fall in [0, 1]

    Keep missing values as -1
    Only use train data to normalize!
    """

    feature_min = train.min(0, keepdims=True)
    feature_range = train.max(0, keepdims=True) - feature_min

    datasets = [train, valid, test]
    # print(datasets[0][:5], datasets[1][:5], datasets[2][:5])

    for i, data in enumerate(datasets):
        missing_idx = data < 0
        data = (data - feature_min) / feature_range
        data[missing_idx] = -1
        datasets[i] = data

    # print(datasets[0][:5], datasets[1][:5], datasets[2][:5])

    return datasets

def split(data, cat_data, y):
    # for support2, 9105 datapoints
    # split into 7105 train, 1000 valid, 1000 test
    # don't shuffle data?
    n_train = 7105
    n_valid = n_train + 1000

    train_data = data[:n_train], cat_data[:n_train], y[:n_train]
    valid_data = data[n_train:n_valid], cat_data[n_train:n_valid], y[n_train:n_valid]
    test_data = data[n_valid:], cat_data[n_valid:], y[n_valid:]

    assert len(train_data[0]) == len(train_data[1]) and len(train_data[1]) == len(train_data[2])
    assert len(valid_data[0]) == len(valid_data[1]) and len(valid_data[1]) == len(valid_data[2])
    assert len(test_data[0]) == len(test_data[1]) and len(test_data[1]) == len(test_data[2])

    print(f"# train: {len(train_data)}")
    print(f"# valid: {len(valid_data)}")
    print(f"# test: {len(test_data)}")

    return train_data, valid_data, test_data

def load_support2(filename, cat_file, y_file, split):
    # data = pd.read_csv(filename, index_col=0)[:7105]

    # TODO: convert raw data to format for input to model
    # for testing: just take age and crea
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader):
            row = [line[k] for k in support2_keys]
            data.append([float(x) if x != '' else -1 for x in row])

    data = np.array(data, dtype='f4')
    if split == 'train':
        data = data[:7105]
    elif split == 'valid':
        data = data[7105:8105]
    else:
        data = data[8105:]

    cat_data = []
    with open(cat_file, 'r') as f:
        for line in f:
            cat_data.append(line.strip().split())

    y = np.loadtxt(y_file, dtype='f4')
    if len(y.shape) == 1:
        y = np.array(y, dtype='f4')

    print(len(data))
    print(len(cat_data))
    print(y.shape)

    print(data[:3, :3])

    return {
        'x': cat_data,
        'c': data,
        'y': y
    }

class Support2(Dataset):
    """
    TODO: doesn't handle x (interpretable attributes) well currently
    """
    def __init__(self, x, c, y, S):
        # self.S = S.astype('f4')
        self.S = S
        self.x = x
        self.y = y
        # self.context = c.astype('f4')
        self.context = c

        # S: n_antes x n
        print(self.S.shape)

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

def load_support2_all(args):
    # load c
    X, _ = load_sup(args['csv'], split=False)

    # load x
    cat_data = []
    with open(cat_file, 'r') as f:
        for line in f:
            cat_data.append(line.strip().split())

    # load y
    y = np.loadtxt(y_file)
    if len(y.shape) == 1:
        y = np.array(y)

    # TODO: do random splits here?

    return {
        'x': cat_data,
        'c': X,
        'y': y
    }

def load_data(args, dataset):
    if dataset == 'support2':
        train_prefix = args['prefix'] + '_train'
        valid_prefix = args['prefix'] + '_valid'
        test_prefix = args['prefix'] + '_test'

        train_data = load_support2(args['raw_file'],
                                    train_prefix + '.tab',
                                    train_prefix + '.Y',
                                    'train')
        valid_data = load_support2(args['raw_file'],
                                    valid_prefix + '.tab',
                                    valid_prefix + '.Y',
                                    'valid')
        test_data = load_support2(args['raw_file'],
                                    test_prefix + '.tab',
                                    test_prefix + '.Y',
                                    'test')

        # check that all lengths are equal
        assert len(set(len(x) for x in train_data.values())) == 1
        assert len(set(len(x) for x in valid_data.values())) == 1
        assert len(set(len(x) for x in test_data.values())) == 1

        # assert len(train_data[0]) == len(train_data[1]) and len(train_data[1]) == len(train_data[2])
        # assert len(valid_data[0]) == len(valid_data[1]) and len(valid_data[1]) == len(valid_data[2])
        # assert len(test_data[0]) == len(test_data[1]) and len(test_data[1]) == len(test_data[2])

        print(f"# train: {len(train_data['x'])}")
        print(f"# valid: {len(valid_data['x'])}")
        print(f"# test: {len(test_data['x'])}")

        # further preprocessing: normalize (using only training data)
        # and split into train, validation and test splits
        train_norm, valid_norm, test_norm = normalize(train_data['c'], valid_data['c'], test_data['c'])

        train_data['c'] = train_norm
        valid_data['c'] = valid_norm
        test_data['c'] = test_norm

        antes = get_freq_itemsets(train_data['x'], train_data['y'], min_support=30, max_lhs=2)

        train_S = build_satisfiability_matrix(train_data['x'], antes)
        valid_S = build_satisfiability_matrix(valid_data['x'], antes)
        test_S = build_satisfiability_matrix(test_data['x'], antes)
        # S, ante_lens, antes = get_freq_itemsets(train_data['x'], train_data['y'], min_support=30)

        train_data['S'] = train_S
        valid_data['S'] = valid_S
        test_data['S'] = test_S

        return train_data, valid_data, test_data, antes


    else:
        raise NotImplementedError