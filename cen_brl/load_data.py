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
        self.S = S.astype('f4')
        self.x = x
        self.y = y
        self.context = c.astype('f4')

        # S: n x n_antes
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

def load_support2_all(args):
    # load c (already scaled!)
    X, Y = load_sup(args['raw_file'], split=False)

    # load x
    cat_file = args['categorical_file']
    cat_data = []
    with open(cat_file, 'r') as f:
        for line in f:
            cat_data.append(line.strip().split())

    # load y
    y_file = args['label_file']
    y = np.loadtxt(y_file)
    if len(y.shape) == 1:
        y = np.array(y)

    return {
        'x': cat_data,
        'c': X,
        'y': y,
        'y2': Y
    }

def load_data(args, dataset):
    if dataset == 'support2':
        data = load_support2_all(args)

        C = data['c']
        X = data['x']
        Y = data['y']
        Y_extra = data['y2']

        # random split
        N_TRAIN = 7105
        N_VALID = 1000
        N_TEST = 1000

        seed = args['seed']
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))

        X = [X[i] for i in order]
        Y = Y[order]
        Y2 = Y_extra[order]
        C = C[order]

        X_train = X[:N_TRAIN]
        Y_train = Y[:N_TRAIN]
        Y2_train = Y2[:N_TRAIN]
        C_train = C[:N_TRAIN]

        X_valid = X[N_TRAIN : N_TRAIN+N_VALID]
        Y_valid = Y[N_TRAIN : N_TRAIN+N_VALID]
        Y2_valid = Y2[N_TRAIN : N_TRAIN+N_VALID]
        C_valid = C[N_TRAIN : N_TRAIN+N_VALID]

        X_test = X[-N_TEST:]
        Y_test = Y[-N_TEST:]
        Y2_test = Y2[-N_TEST:]
        C_test = C[-N_TEST:]

        # get antecedents from train data
        min_support = args['min_support']
        max_lhs = args['max_lhs']
        antes = get_freq_itemsets(X_train, Y_train, min_support=min_support, max_lhs=max_lhs)

        # get satisfiability matrices
        S_train = build_satisfiability_matrix(X_train, antes)
        S_valid = build_satisfiability_matrix(X_valid, antes)
        S_test = build_satisfiability_matrix(X_test, antes)

        train_data = {
            'x': X_train,
            'y': Y_train,
            'c': C_train,
            'S': S_train,
            'y2': Y2_train,
        }
        valid_data = {
            'x': X_valid,
            'y': Y_valid,
            'c': C_valid,
            'S': S_valid,
            'y2': Y2_valid,
        }
        test_data = {
            'x': X_test,
            'y': Y_test,
            'c': C_test,
            'S': S_test,
            'y2': Y2_test,
        }

        return train_data, valid_data, test_data, antes

    else:
        raise NotImplementedError