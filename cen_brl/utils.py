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

        print(len(data_pos))
        print(len(data_neg))
        print(len(data))
        assert len(data_pos) + len(data_neg) == len(data)

        itemsets = [r[0] for r in fpgrowth(data_pos, supp=min_support, zmax=max_lhs)]
        itemsets.extend([r[0] for r in fpgrowth(data_neg, supp=min_support, zmax=max_lhs)])
    else:
        raise NotImplementedError

    itemsets = list(set(itemsets))
    print("{} rules mined".format(len(itemsets)))

    # build S (antecedent vs. datapoint matrix)
    # S[i] is the i-th antecedent
    # S[0] is for the default rule (which satisfies all data)

    print("Building S...")
    """
    S = [set() for _ in range(len(itemsets) + 1)]
    S[0] = set(range(len(data)))

    for j, lhs in enumerate(itemsets):
        s_lhs = set(lhs)
        S[j+1] = set([i for i, xi in enumerate(data) if s_lhs.issubset(xi)])
    """

    n_antes = len(itemsets)

    S = np.zeros((n_antes + 1, len(data)))
    S[0] = 1.
    for j, lhs in enumerate(itemsets):
        s_lhs = set(lhs)
        for i, xi in enumerate(data):
            S[j+1, i] = s_lhs.issubset(xi)

    S = S.transpose()
    print("S built.")

    # get the cardinality of each antecendent
    # default rule has cardinality 0
    lhs_len = [0]
    lhs_len.extend([len(lhs) for lhs in itemsets])

    lhs_len = np.array(lhs_len)
    itemsets = ['null'] + itemsets
    
    return S, lhs_len, itemsets
