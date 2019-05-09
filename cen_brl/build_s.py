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
    antes = [set(x) for x in antes]
    data = [set(x) for x in data]
    # itertools.product(antes, [data])
    with multiprocessing.Pool(4) as pool:
        S = pool.starmap(get_row, itertools.product(antes, [data]))
    
    S = np.array(S, dtype=np.bool)
    print(S.shape)
    return S

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_init({'raw_file': sys.argv[1]})

    train_features, test_features = get_interpretable_features({
        'interp_file': sys.argv[2],
        'interp_type': sys.argv[3]
    })

    min_support = int(sys.argv[4])
    max_lhs = int(sys.argv[5])

    antes = get_freq_itemsets(train_features, y_train, min_support=min_support, max_lhs=max_lhs)

    # count number of features that appear in all
    antes_mapping = {
        i: a for i, a in enumerate(antes)
    }
    print(antes_mapping[0])
    print(antes_mapping[1])
    print(len(antes))

    st = time.time()
    S_train = build_sat_parallel(train_features, antes)
    # S_train = build_satisfiability_matrix(train_features, antes)
    print(time.time() - st)
    print(S_train.shape)

    print(S_train.sum(0).max())
    print(S_train.sum(0).min())
    print(np.percentile(S_train.sum(0), [10, 30, 50, 70, 90]))

    # S_test = build_satisfiability_matrix(test_features, antes)

    # save matrices
    # np.savez(sys.argv[6], S_train=S_train, S_test=S_test)