"""
Load and preprocess MNIST data.

Based on CEN implementation
"""

import numpy as np
from torch.utils.data import Dataset

from utils import get_freq_itemsets, build_satisfiability_matrix

class MNIST(Dataset):
    def __init__(self, x, c, y, S):
        self.S = S.astype('f4')
        self.x = x
        self.y = y
        self.context = c.astype('f4')

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

def get_interpretable_features(x, width=4):
    """
    Returns list of superpixel values as interpretable features.
    """

def load_mnist_init(args):
    """
    Initial MNIST loading. Just renormalizes images, reshapes
    data for CNN, and changes targets to one-hot
    """

    f = np.load(args['raw_file'])
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']

    # change to floats and reshape
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # change from labels to one-hot
    y_train_new = np.zeros((len(y_train), 10))
    y_test_new = np.zeros((len(y_test), 10))

    y_train_new[np.arange(len(y_train)), y_train] = 1
    y_test_new[np.arange(len(y_test)), y_test] = 1
    y_train = y_train_new
    y_test = y_test_new

    return (x_train, y_train), (x_test, y_test)

def load_mnist(args):
    (x_train, y_train), (x_test, y_test) = load_mnist_init(args)

    # get interpretable features: 4x4 superpixels?
    train_features = get_interpretable_features(x_train)
    test_features = get_interpretable_features(x_test)

    # split into train and validation
    N_TRAIN = len(x_train)
    N_VALID = int(0.1 * N_TRAIN)
    N_TRAIN -= N_VALID

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
    }