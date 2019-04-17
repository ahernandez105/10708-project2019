import csv

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from utils import get_freq_itemsets


def load_support2(filename, cat_file, y_file):
    # data = pd.read_csv(filename, index_col=0)[:7105]

    # TODO: convert raw data to format for input to model
    # for testing: just take age and crea
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader):
            row = [line['age'], line['temp']]
            data.append([float(x) if x != '' else -1 for x in row])
            
            if i >= 7104:
                break

    data = np.array(data)

    cat_data = []
    with open(cat_file, 'r') as f:
        for line in f:
            cat_data.append(line.strip().split())

    y = np.loadtxt(y_file, dtype=int)
    if len(y.shape) == 1:
        y = np.array(y)

    print(len(data))
    print(len(cat_data))
    print(y.shape)

    print(data[:5])

    return cat_data, data, y

class Support2(Dataset):
    def __init__(self, x, c, y, S):
        self.S = S.astype('f4')
        self.x = x
        self.y = y
        self.context = c.astype('f4')

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

def load_data(args, dataset):
    if dataset == 'support2':
        x, c, y = load_support2(args['raw_file'], args['categorical_file'],
                        args['label_file'])
        S, ante_lens, antes = get_freq_itemsets(x, y)

        return x, c, y, S, antes

    else:
        raise NotImplementedError