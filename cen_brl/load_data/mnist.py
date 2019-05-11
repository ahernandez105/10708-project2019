"""
Load and preprocess MNIST data.

Based on CEN implementation
"""

import json
import sys
import time

import numpy as np
import scipy as sp
from torch.utils.data import Dataset
from skimage.feature import hog
from skimage.transform import resize
from skimage import io
from PIL import Image 

from utils import get_freq_itemsets, build_satisfiability_matrix

"""
From CEN
"""
def get_zca_whitening_mat(X, eps=1e-6):
    flat_X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
    Sigma = np.dot(flat_X.T, flat_X) / flat_X.shape[0]
    U, S, _ = sp.linalg.svd(Sigma)
    M = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + eps))), U.T)
    return M

def zca_whiten(X, W):
    shape = X.shape
    flat_X = np.reshape(X, (shape[0], np.prod(shape[1:])))
    white_X = np.dot(flat_X, W)
    return np.reshape(white_X, shape)

class MNIST(Dataset):
    def __init__(self, data):
        y = data['y']
        S = data['S']
        c = data['c']

        self.S = S.astype('f4')
        # self.x = x
        self.y = y
        self.context = np.transpose(c.astype('f4'), [0, 3, 1, 2])

        self.S = self.S / self.S.sum(0, keepdims=True)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # return the corresponding row of S, context, x, and label.

        return {
            'S': self.S[idx],
            'context': self.context[idx],
            # 'x': self.x[idx],
            'y': self.y[idx],
        }

def get_interpretable_features(args):
    """
    """
    z_train, z_test = load_interp_features(datapath=args['interp_file'],
                                           feature_type=args['interp_type'],
                                           remove_const_features=False,
                                           standardize=False,
                                           whiten=False,
                                           permute=False)

    # superpixel antecedents: either pixel >= 0.5, or pixel < 0.5
    c_train = [
        [f'({i}, {j}) >= 0.5' if c else f'({i}, {j}) < 0.5'
        for i, r in enumerate(z) for j, c in enumerate(r)]
        for z in (z_train >= 0.5)
    ]
    c_test = [
        [f'({i}, {j}) >= 0.5' if c else f'({i}, {j}) < 0.5'
        for i, r in enumerate(z) for j, c in enumerate(r)]
        for z in (z_test >= 0.5)
    ]

    return (c_train, c_test), (z_train, z_test)

def load_interp_features(datapath=None,
                         feature_type='pixels16x16',
                         feature_subset_per=None,
                         remove_const_features=True,
                         standardize=True,
                         whiten=False,
                         permute=True,
                         signal_to_noise=None,
                         seed=42,
                         verbose=1):
    """Load an interpretable representation for MNIST.

    Args:
        datapath: str or None (default: None)
        feature_type: str (default: 'pixels16x16')
            Possible values are:
            {'pixels16x16', 'pixels20x20', 'pixels28x28', 'hog3x3'}.
        standardize: bool (default: True)
        whiten: bool (default: False)
        permute: bool (default: True)
        signal_to_noise: float or None (default: None)
            If not None, adds white noise to each feature with a specified SNR.
        seed: uint (default: 42)
        verbose: uint (default: 1)

    Returns:
        data: tuple of (Z_train, Z_test) ndarrays
    """
    # if datapath is None:
    #     datapath = "$DATA_PATH/MNIST/mnist.interp.%s.npz" % feature_type
    # datapath = os.path.expandvars(datapath)

    if verbose:
        print("Loading interpretable features...")

    data = np.load(datapath)
    Z_train, Z_test = data['Z_train'], data['Z_test']

    if feature_type.startswith('pixels'):
        Z_train = Z_train.astype('float32')
        Z_test = Z_test.astype('float32')
        Z_train /= 255
        Z_test /= 255

        _, h, w = Z_train.shape

    n_train = Z_train.shape[0]
    n_test = Z_test.shape[0]

    Z_train = Z_train.reshape((n_train, -1))
    Z_test = Z_test.reshape((n_test, -1))

    if remove_const_features:
        Z_std = Z_train.std(axis=0)
        nonconst = np.where(Z_std > 1e-5)[0]
        Z_train = Z_train[:, nonconst]
        Z_test = Z_test[:, nonconst]

    if standardize:
        Z_mean = Z_train.mean(axis=0)
        Z_std = Z_train.std(axis=0)
        nonconst = np.where(Z_std > 1e-5)[0]
        Z_train -= Z_mean
        Z_train[:, nonconst] /= Z_std[nonconst]
        Z_test -= Z_mean
        Z_test[:, nonconst] /= Z_std[nonconst]

    if whiten:
        WM = get_zca_whitening_mat(Z_train)
        Z_train = zca_whiten(Z_train, WM)
        Z_test = zca_whiten(Z_test, WM)

    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(Z_train))
        Z_train = Z_train[order]

    if feature_subset_per is not None:
        assert feature_subset_per > 0. and feature_subset_per <= 1.
        feature_subset_size = int(Z_train.shape[1] * feature_subset_per)
        rng = np.random.RandomState(seed)
        feature_idx = rng.choice(Z_train.shape[1],
                                 size=feature_subset_size,
                                 replace=False)
        Z_train = Z_train[:, feature_idx]
        Z_test = Z_test[:, feature_idx]

    if signal_to_noise is not None and signal_to_noise > 0.:
        rng = np.random.RandomState(seed)
        N_train = np.random.normal(scale=1./signal_to_noise,
                                   size=Z_train.shape)
        N_test = np.random.normal(scale=1./signal_to_noise,
                                  size=Z_test.shape)
        Z_train += N_train
        Z_test += N_test

    if feature_type.startswith('pixels'):
        Z_train = Z_train.reshape((n_train, h, w))
        Z_test = Z_test.reshape((n_test, h, w))

    return Z_train, Z_test

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

    # split into train and validation
    N_TRAIN = 50000
    N_VALID = 10000

    seed = args['seed']
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(x_train))

    Y = y_train[order]
    C = x_train[order]

    Y = y_train[order]
    C = x_train[order]

    y_train = Y[:N_TRAIN]
    c_train = C[:N_TRAIN]

    y_valid = Y[N_TRAIN:]
    c_valid = C[N_TRAIN:]

    c_test = x_test

    train_data = {
        'y': y_train,
        'c': c_train,
        # 'S': S_train,
    }
    valid_data = {
        'y': y_valid,
        'c': c_valid,
        # 'S': S_valid,
    }
    test_data = {
        'y': y_test,
        'c': c_test,
        # 'S': S_test,
    }


    # load interpretable features
    if args['categorical_file']:
        x = np.load(args['categorical_file'])
        S = x['S_train'][order]

        S_train = S[:N_TRAIN]
        S_valid = S[N_TRAIN:]
        S_test = x['S_test']

        # load antecedents
        antes = json.load(open(args['ante_file'], 'r'))
        antes = [antes[str(i)] for i in range(len(antes))]

    else:
        (train_features, test_features), _ = get_interpretable_features(args)

        X = [train_features[i] for i in order]

        x_train = X[:N_TRAIN]
        x_valid = X[N_TRAIN:]
        x_test = test_features

        # get antecedents from train data
        min_support = args['min_support']
        max_lhs = args['max_lhs']
        antes = get_freq_itemsets(x_train, y_train, min_support=min_support, max_lhs=max_lhs)

        # build satisfiability matrix
        S_train = build_satisfiability_matrix(x_train, antes)
        S_valid = build_satisfiability_matrix(x_valid, antes)
        S_test = build_satisfiability_matrix(x_test, antes)

        train_data['x'] = x_train
        valid_data['x'] = x_valid
        test_data['x'] = x_test

    train_data['S'] = S_train
    valid_data['S'] = S_valid
    test_data['S'] = S_test

    return {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'antes': antes,
    }

if __name__ == '__main__':
    mnist_file = sys.argv[1]

    f = np.load(mnist_file)
    x_train = f['x_train']
    x_test = f['x_test']

    # uint_feat = get_interpretable_features(x_train)
    # float_feat = get_interpretable_features(x_train.astype('float32') / 255)

    print(np.allclose(uint_feat, float_feat))
