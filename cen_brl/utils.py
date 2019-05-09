import sys

import numpy as np
from fim import fpgrowth

def get_freq_itemsets(data, y, min_support=50, max_lhs=2):
    """
    Xtrain,Ytrain,nruleslen,lhs_len,itemsets = get_freqitemsets(fname+'_train',minsupport,maxlhs) #Do frequent itemset mining from the training data
    """

    if y.shape[-1] == 2:
        # currently only mine itemsets for binary classification
        data_pos = [x for i, x in enumerate(data) if y[i, 0] == 0]
        data_neg = [x for i, x in enumerate(data) if y[i, 0] == 1]

        print("# positive class:", len(data_pos))
        print("# negative class:", len(data_neg))
        print(len(data))
        assert len(data_pos) + len(data_neg) == len(data)

        itemsets = [r[0] for r in fpgrowth(data_pos, supp=min_support, zmax=max_lhs)]
        itemsets.extend([r[0] for r in fpgrowth(data_neg, supp=min_support, zmax=max_lhs)])
    else:
        data_classes = [[] for _ in range(y.shape[-1])]
        classes = y.argmax(-1)
        for i, c in enumerate(classes):
            data_classes[c].append(data[i])

        print([len(dc) for dc in data_classes])
        # for i, dc in enumerate(data_classes):
        #     print(f"# {i}: {len(dc)}")

        assert sum([len(dc) for dc in data_classes]) == len(data)

        itemsets = [
            [r[0] for r in fpgrowth(data_class, supp=min_support, zmax=max_lhs)]
            for data_class in data_classes
        ]
        # flatten
        itemsets = [x for class_itemset in itemsets for x in class_itemset]

    itemsets = list(set(itemsets))
    print("{} rules mined".format(len(itemsets)))

    return itemsets

    # itemsets = [('null',)] + itemsets

    # build S (antecedent vs. datapoint matrix)
    # S[i] is the i-th antecedent
    # S[0] is for the default rule (which satisfies all data)

    print("Building S...")

    S = build_satisfiability_matrix(data, itemsets)

    print(S.shape)
    print("S built.")

    return itemsets

def build_satisfiability_matrix(data, antes, prefix=None):
    """
    Build S
    S[i,j] = 1 if datapoint x_i satisfies antecendent j,
        but doesn't satisfy the antecedents in prefix.
        0 otherwise.
    The null rule is not included in S, but it is always satisfied.

    data: list of datapoints
    antes: list of antecedents
    prefix: list of indices of previous antecedents in current decision list
    """

    data = [set(xi) for xi in data]
    antes = [set(lhs) for lhs in antes]
    n_antes = len(antes)

    print(f"# antes: {n_antes}")

    if prefix:
        prefix = [set(antes[lhs]) for lhs in prefix]

        prefix_satisfied = np.zeros((len(data),), dtype=np.bool)
        for i, xi in enumerate(data):
            prefix_sat = False
            for a in prefix:
                if a.issubset(xi):
                    prefix_sat = True
                    break

            prefix_satisfied[i] = prefix_sat
    else:
        prefix_satisfied = np.zeros((len(data),), dtype=np.bool)

    # S = np.zeros((len(data), n_antes), dtype='f4')
    S = np.zeros((len(data), n_antes), dtype=np.bool)

    # True: 1, False: 0
    # S[0] = np.logical_not(prefix_satisfied).astype(np.float)
    for j, lhs in enumerate(antes):
        if 'null' in lhs:
            # ignore the null rule
            continue

        for i, xi in enumerate(data):
            if prefix_satisfied[i]:
                continue
                
            S[i, j] = lhs.issubset(xi)

        if j % 20 == 0:
            print(j, end='..')
            sys.stdout.flush()

    print(S.shape)

    return S
