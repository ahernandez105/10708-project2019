import csv
import os
import sys

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# from utils import get_freq_itemsets, build_satisfiability_matrix
# from support2_mod import load_data as load_sup
from .support2 import load_support2
from .imdb import load_imdb

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

def load_data(args, dataset):
    if dataset == 'support2':
        return load_support2(args)

    elif dataset == 'imdb':
        return load_imdb(args)

    else:
        raise NotImplementedError