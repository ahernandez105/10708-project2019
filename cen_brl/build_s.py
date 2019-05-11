import sys
import multiprocessing
import itertools
import time
import json

import numpy as np

from utils import get_freq_itemsets
from load_data.mnist import load_mnist_init, get_interpretable_features

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

def build_sat(z_train, z_test, antes, n=10000):
    """
    Find top n antecedents to build S
    """

    n = min(n, len(antes))

    n_antes = len(antes)
    n_train = len(z_train)
    n_test = len(z_test)

    # mapping from ante idx to satisfy count
    satisfy_counts = {}

    a1 = (z_train >= 0.5)
    a2 = (z_train < 0.5)
    b1 = (z_test >= 0.5)
    b2 = (z_test < 0.5)

    for i, ante in enumerate(antes):
        satisfy = np.full((n_train,), True, dtype=np.bool)

        for clause in ante:
            x, y, sign, _ = clause.split()
            x = int(x[1:-1])
            y = int(y[:-1])

            if sign == '>=':
                satisfy = np.logical_and(satisfy, a1[:, x, y])
            else:
                satisfy = np.logical_and(satisfy, a2[:, x, y])

        n_satisfy = np.count_nonzero(satisfy)
        satisfy_counts[i] = n_satisfy

        if i % 2000 == 0:
            print(i, end='..')
            sys.stdout.flush()
    print()

    top_n = sorted(satisfy_counts.items(), key=lambda x: x[1])[:n]

    S_train = np.full((n, n_train), True, dtype=np.bool)
    S_test = np.full((n, n_test), True, dtype=np.bool)

    filtered_antes = [antes[idx] for idx, _ in top_n]

    for i, ante in enumerate(filtered_antes):
        for clause in ante:
            x, y, sign, _ = clause.split()
            x = int(x[1:-1])
            y = int(y[:-1])

            if sign == '>=':
                S_train[i] = np.logical_and(S_train[i], a1[:, x, y])
                S_test[i] = np.logical_and(S_test[i], b1[:, x, y])
            else:
                S_train[i] = np.logical_and(S_train[i], a2[:, x, y])
                S_test[i] = np.logical_and(S_test[i], b2[:, x, y])

        if i % 2000 == 0:
            print(i, end='..')
            sys.stdout.flush()
    print()

    # return S_train.transpose(), S_test.transpose(), antes
    return S_train.transpose(), S_test.transpose(), filtered_antes

def filter_top(S_train, S_test, n=10000):
    """
    Only use the n antecedents that satisfy the fewest training examples,
    to save memory
    """
    
    n = min(n, S_train.shape[-1])

    c = S_train.sum(0)
    idxes = np.argsort(c)[:n]

    S_train_sub = S_train[:, idxes]
    S_test_sub = S_test[:, idxes]

    # check that all datapoints satisfy at least 1 antecedent?
    print(len(S_train_sub) - np.count_nonzero(S_train_sub.sum(-1)))
    print(len(S_test_sub) - np.count_nonzero(S_test_sub.sum(-1)))

    return S_train_sub, S_test_sub, idxes

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_init({'raw_file': sys.argv[1]})

    (train_features, test_features), (z_train, z_test) = get_interpretable_features({
        'interp_file': sys.argv[2],
        'interp_type': sys.argv[3]
    })

    min_support = int(sys.argv[4])
    max_lhs = int(sys.argv[5])

    antes = get_freq_itemsets(train_features, y_train, min_support=min_support, max_lhs=max_lhs)

    st = time.time()
    if len(sys.argv) > 8:
        n = int(sys.argv[8])
    else:
        n = 10000
    S_train, S_test, antes_sub = build_sat(z_train, z_test, antes, n=n)

    assert S_train.shape[-1] == len(antes_sub)
    assert S_test.shape[-1] == len(antes_sub)
    print(time.time() - st)

    print(S_train.shape)
    print(S_test.shape)

    # check that all datapoints satisfy at least 1 antecedent?
    print(len(S_train) - np.count_nonzero(S_train.sum(-1)))
    print(len(S_test) - np.count_nonzero(S_test.sum(-1)))

    # S_train = build_sat_quick(z_train, antes)
    # print()
    # S_test = build_sat_quick(z_test, antes)
    # print()
    # 
    # S_train, S_test, idxes = filter_top(S_train, S_test, n=10000)

    # antes = [antes[i] for i in idxes]

    # save matrices
    np.savez(sys.argv[6], S_train=S_train, S_test=S_test)

    # save antecedents
    json.dump({i: ante for i, ante in enumerate(antes_sub)}, open(sys.argv[7], 'w'))
