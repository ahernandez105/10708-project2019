import sys
import multiprocessing
import itertools
import time

import numpy as np

from utils import get_freq_itemsets, build_satisfiability_matrix
from load_data.mnist import load_mnist_init, get_interpretable_features

def get_row(lhs, data):
    # lhs = antes[j]
    return [lhs.issubset(xi) for xi in data]

def build_sat_parallel(data, antes):
    # itertools.product(antes, [data])
    with multiprocessing.Pool(4) as pool:
        S = pool.starmap(get_row, itertools.product(antes, [data]))
    
    S = np.array(S, dtype=np.bool)
    print(S.shape)
    return S

def build_sat_quick(z, antes):
    n_antes = len(antes)
    n_points = len(z)

    S = np.full((n_antes, n_points), True, dtype=np.bool)

    a1 = (z >= 0.5)
    a2 = (z < 0.5)

    for i, ante in enumerate(antes):
        for clause in ante:
            x, y, sign, _ = clause.split()
            x = int(x[1:-1])
            y = int(y[:-1])

            if sign == '>=':
                S[i] = np.logical_and(S[i], a1[:, x, y])
            else:
                S[i] = np.logical_and(S[i], a2[:, x, y])

        if i % 1000 == 0:
            print(i, end='..')

    return S.transpose()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_init({'raw_file': sys.argv[1]})

    (train_features, test_features), (z_train, z_test) = get_interpretable_features({
        'interp_file': sys.argv[2],
        'interp_type': sys.argv[3]
    })

    min_support = int(sys.argv[4])
    max_lhs = int(sys.argv[5])

    antes = get_freq_itemsets(train_features, y_train, min_support=min_support, max_lhs=max_lhs)

    S_train = build_sat_quick(z_train, antes)
    print()
    S_test = build_sat_quick(z_test, antes)
    
    # save matrices
    np.savez(sys.argv[6], S_train=S_train, S_test=S_test)