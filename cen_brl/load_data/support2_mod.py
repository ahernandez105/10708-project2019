"""Loaders and preprocessors for SUPPORT2 data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six

import numpy as np
import pandas as pd

# Data parameters
TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 7105, 1000, 1000
EXCLUDE_FEATURES = [
    'aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'dnr', 'dnrday',
    'hospdead', 'dzclass', 'edu', 'scoma', 'totmcst', 'charges', 'totcst',
]
TARGETS = ['death', 'd.time']
AVG_VALUES = {
    'alb':      3.5,
    'bili':     1.01,
    'bun':      6.51,
    'crea':     1.01,
    'pafi':     333.3,
    'wblc':     9.,
    'urine':    2502.,
}

def load_data(datapath=None,
              nb_intervals=156,
              interval_len=7,
              fill_na='avg',
              na_value=0.0,
              death_indicator=1.0,
              censorship_indicator=1.0,
              inputs_as_sequences=False,
              inputs_pad_mode='constant',
              verbose=True,
              seed=42,
              split=False):
    """Load and preprocess the SUPPORT2 dataset.

    Args:
        datapath : str or None
        nb_intervals : uint (default: 100)
            Number of intervals to split the time line.
        interval_len : uint (default: 20)
            The length of the interval in days.
        fill_na : str (default: 'avg')
        na_value : float (default: -1.0)
        death_indicator : float (default: 1.0)
        censorship_indicator : float (default: -1.0)
        verbose : bool (default: True)
        seed : uint (default: 42)

    Returns:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """
    if datapath is None:
        datapath = '$DATA_PATH/SUPPORT2/support2.csv'
    datapath = os.path.expandvars(datapath)

    if verbose:
        print("Loading data...")

    # Exclude unnecessary columns
    df = pd.read_csv(datapath)
    columns = sorted(list(set(df.columns) - set(EXCLUDE_FEATURES)))
    df = df[columns]

    # Split into features and targets
    targets = df[TARGETS]
    features = df[list(set(df.columns) - set(TARGETS))]

    # Convert categorical columns into one-hot format
    cat_columns = features.columns[features.dtypes == 'object']
    features = pd.get_dummies(features, dummy_na=False, columns=cat_columns)

    # Scale and impute real-valued features
    features_orig = features.copy(deep=True)
    # impute before scaling!!
    if fill_na == 'avg':
        for key, val in six.iteritems(AVG_VALUES):
            features[[key]] = features[[key]].fillna(val)

    features[['num.co', 'slos', 'hday']] = \
        features[['num.co', 'slos', 'hday']].astype(np.float)
    float_cols = features.columns[features.dtypes == np.float]
    features[float_cols] = \
        (features[float_cols] - features[float_cols].min()) / \
        (features[float_cols].max() - features[float_cols].min())
    features.fillna(na_value, inplace=True)
    X = features.values
    # X[:, 33] = np.random.rand(X.shape[0])

    # Preprocess targets
    T = targets.values
    Y = np.zeros((len(targets), nb_intervals, 2))
    for i, (death, days) in enumerate(T):
        intervals = days // interval_len
        if death and intervals < nb_intervals:
            Y[i, intervals:, 1] = death_indicator
        if not death and intervals < nb_intervals:
            Y[i, intervals:, 0] = censorship_indicator

    # Convert inputs into sequences if necessary
    if inputs_as_sequences:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        if inputs_pad_mode == 'constant':
            X = np.pad(X, [(0, 0), (0, Y.shape[1] - 1), (0, 0)],
                       mode=inputs_pad_mode,
                       constant_values=0.0)
        else:
            X = np.pad(X, [(0, 0), (0, Y.shape[1] - 1), (0, 0)],
                       mode=inputs_pad_mode)

    if split:
        return split_data(X, Y, seed=seed)
    else:
        return X, Y, features.columns, features_orig.values

def split_data(X, Y, seed=42):
    # Shuffle & split the data into sets
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(X))
    X, Y = X[order], Y[order]

    X_train = X[:TRAIN_SIZE]
    y_train = Y[:TRAIN_SIZE]
    X_valid = X[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE]
    y_valid = Y[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE]
    X_test = X[-TEST_SIZE:]
    y_test = Y[-TEST_SIZE:]

    if verbose:
        print('X shape:', X.shape)
        print('Y shape:', Y.shape)
        print(len(X_train), 'train instances')
        print(len(X_valid), 'validation instances')
        print(len(X_test), 'test instances')

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_data_orig(datapath=None,
              nb_intervals=156,
              interval_len=7,
              fill_na='avg',
              na_value=0.0,
              death_indicator=1.0,
              censorship_indicator=1.0,
              inputs_as_sequences=False,
              inputs_pad_mode='constant',
              verbose=True,
              seed=42):
    """Load and preprocess the SUPPORT2 dataset.

    Args:
        datapath : str or None
        nb_intervals : uint (default: 100)
            Number of intervals to split the time line.
        interval_len : uint (default: 20)
            The length of the interval in days.
        fill_na : str (default: 'avg')
        na_value : float (default: -1.0)
        death_indicator : float (default: 1.0)
        censorship_indicator : float (default: -1.0)
        verbose : bool (default: True)
        seed : uint (default: 42)

    Returns:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """
    if datapath is None:
        datapath = '$DATA_PATH/SUPPORT2/support2.csv'
    datapath = os.path.expandvars(datapath)

    if verbose:
        print("Loading data...")

    # Exclude unnecessary columns
    df = pd.read_csv(datapath)
    columns = sorted(list(set(df.columns) - set(EXCLUDE_FEATURES)))
    df = df[columns]

    # Split into features and targets
    targets = df[TARGETS]
    features = df[list(set(df.columns) - set(TARGETS))]

    # Convert categorical columns into one-hot format
    cat_columns = features.columns[features.dtypes == 'object']
    features = pd.get_dummies(features, dummy_na=False, columns=cat_columns)

    # Scale and impute real-valued features
    features[['num.co', 'slos', 'hday']] = \
        features[['num.co', 'slos', 'hday']].astype(np.float)
    float_cols = features.columns[features.dtypes == np.float]
    features[float_cols] = \
        (features[float_cols] - features[float_cols].min()) / \
        (features[float_cols].max() - features[float_cols].min())
    if fill_na == 'avg':
        for key, val in six.iteritems(AVG_VALUES):
            features[[key]] = features[[key]].fillna(val)
    features.fillna(na_value, inplace=True)
    X = features.values
    X[:, 33] = np.random.rand(X.shape[0])

    # Preprocess targets
    T = targets.values
    Y = np.zeros((len(targets), nb_intervals, 2))
    for i, (death, days) in enumerate(T):
        intervals = days // interval_len
        if death and intervals < nb_intervals:
            Y[i, intervals:, 1] = death_indicator
        if not death and intervals < nb_intervals:
            Y[i, intervals:, 0] = censorship_indicator

    # Convert inputs into sequences if necessary
    if inputs_as_sequences:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        if inputs_pad_mode == 'constant':
            X = np.pad(X, [(0, 0), (0, Y.shape[1] - 1), (0, 0)],
                       mode=inputs_pad_mode,
                       constant_values=0.0)
        else:
            X = np.pad(X, [(0, 0), (0, Y.shape[1] - 1), (0, 0)],
                       mode=inputs_pad_mode)

    # Shuffle & split the data into sets
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(X))
    X, Y = X[order], Y[order]

    X_train = X[:TRAIN_SIZE]
    y_train = Y[:TRAIN_SIZE]
    X_valid = X[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE]
    y_valid = Y[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE]
    X_test = X[-TEST_SIZE:]
    y_test = Y[-TEST_SIZE:]

    if verbose:
        print('X shape:', X.shape)
        print('Y shape:', Y.shape)
        print(len(X_train), 'train instances')
        print(len(X_valid), 'validation instances')
        print(len(X_test), 'test instances')

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
