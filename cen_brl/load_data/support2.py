import numpy as np

from torch.utils.data import Dataset

from .support2_mod import load_data as load_sup
from utils import get_freq_itemsets, build_satisfiability_matrix

support2_keys = [
    'age',
    'temp',
    'crea',
    'sod',
    'hrt',
    'resp',
    'meanbp',
    'wblc',
    'sps',
    'surv2m',
    'surv6m',
    'ph',
    'pafi',
    'alb',
    'bili',
]

class Support2(Dataset):
    """
    TODO: doesn't handle x (interpretable attributes) well currently
    """
    def __init__(self, x, c, y, S):
        self.S = S.astype('f4')
        self.x = x
        self.y = y
        self.context = c.astype('f4')

        # S: n x n_antes
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

def make_antecedents(X, feat_names, orig_values, ignore_missing=True):
    categorical = set([
       'sex_female', 'sex_male', 'income_$11-$25k', 'income_$25-$50k',
       'income_>$50k', 'income_under $11k', 'sfdm2_<2 mo. follow-up',
       'sfdm2_Coma or Intub', 'sfdm2_SIP>=30', 'sfdm2_adl>=4 (>=5 if sur)',
       'sfdm2_no(M2 and SIP pres)', 'ca_metastatic', 'ca_no', 'ca_yes',
       'dzgroup_ARF/MOSF w/Sepsis', 'dzgroup_CHF', 'dzgroup_COPD',
       'dzgroup_Cirrhosis', 'dzgroup_Colon Cancer', 'dzgroup_Coma',
       'dzgroup_Lung Cancer', 'dzgroup_MOSF w/Malig', 'race_asian',
       'race_black', 'race_hispanic', 'race_other', 'race_white',
       'dementia', 'diabetes'
    ])
    real_feats = set(feat_names).difference(categorical)

    idx_mapping = {
        feat: i for i, feat in enumerate(feat_names)
    }

    # compute percentiles (for now just on whole dataset, not just train)
    percentiles = {}
    for feat in real_feats:
        vals = np.percentile(X[:, idx_mapping[feat]], [25, 50, 75])
        percentiles[feat] = vals

    features = []

    assert X.shape == orig_values.shape
    for row, o_row in zip(X, orig_values):
        row_feats = []
        for feat, i in idx_mapping.items():
            if ignore_missing and np.isnan(o_row[i]):
                # ignore missing values
                # print('ignore')
                continue

            if feat in categorical:
                if row[i] > 0:
                    row_feats.append(feat)
            
            else:
                # p25, p50, p75 = percentiles[feat]

                done = False
                for i, p in enumerate(percentiles[feat]):
                    if p == 0:
                        continue

                    if row[i] < p:
                        done = True
                        if i == 0:
                            row_feats.append(f"{feat}<{p:.3f}")
                        else:
                            below = percentiles[feat][i-1]
                            row_feats.append(f"{below:.3f}<={feat}<{p:.3f}")

                if not done:
                    assert row[i] >= percentiles[feat][-1]
                    row_feats.append(f"{feat}>={percentiles[feat][-1]:.3f}")
                            
                # bool_25 = (row[i] < p25)
                # bool_50 = (row[i] < p50)
                # bool_75 = (row[i] < p75)
                # # bool_25 = (X[:, i] < p25)
                # # bool_50 = (X[:, i] < p50)
                # # bool_75 = (X[:, i] < p75)

                # for cond, val in [(bool_25, p25), (bool_50, p50), (bool_75, p75)]:
                #     if val == 0:
                #         continue
                #     if cond:
                #         row_feats.append(f"{feat}<{val:.3f}")
                #     else:
                #         row_feats.append(f"{feat}>={val:.3f}")

        features.append(list(set(row_feats)))

    return features

def load_support2_all(args):
    # load c (already scaled!)
    X, Y, feat_names, orig_values = load_sup(args['raw_file'], split=False)

    # convert C to antecedents (x)
    if args['categorical_file']:
        # load (old) x
        cat_file = args['categorical_file']
        cat_data = []
        with open(cat_file, 'r') as f:
            for line in f:
                cat_data.append(line.strip().split())
    else:
        cat_data = make_antecedents(X, feat_names, orig_values, ignore_missing=(not args['use_missing']))

    # load y
    y_file = args['label_file']
    y = np.loadtxt(y_file)
    if len(y.shape) == 1:
        y = np.array(y)

    return {
        'x': cat_data,
        'c': X,
        'y': y,
        'y2': Y
    }

def load_support2(args):
    data = load_support2_all(args)

    C = data['c']
    X = data['x']
    Y = data['y']
    Y_extra = data['y2']

    N_TRAIN = 7105
    N_VALID = 1000
    N_TEST = 1000

    if args['order']:
        order = np.load(args['order'])
    else:
        # random split
        print("random split...")
        seed = args['seed']
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))

    X = [X[i] for i in order]
    Y = Y[order]
    Y2 = Y_extra[order]
    C = C[order]

    X_train = X[:N_TRAIN]
    Y_train = Y[:N_TRAIN]
    Y2_train = Y2[:N_TRAIN]
    C_train = C[:N_TRAIN]

    X_valid = X[N_TRAIN : N_TRAIN+N_VALID]
    Y_valid = Y[N_TRAIN : N_TRAIN+N_VALID]
    Y2_valid = Y2[N_TRAIN : N_TRAIN+N_VALID]
    C_valid = C[N_TRAIN : N_TRAIN+N_VALID]

    X_test = X[-N_TEST:]
    Y_test = Y[-N_TEST:]
    Y2_test = Y2[-N_TEST:]
    C_test = C[-N_TEST:]

    # get antecedents from train data
    min_support = args['min_support']
    max_lhs = args['max_lhs']
    antes = get_freq_itemsets(X_train, Y_train, min_support=min_support, max_lhs=max_lhs)

    # get satisfiability matrices
    S_train = build_satisfiability_matrix(X_train, antes)
    S_valid = build_satisfiability_matrix(X_valid, antes)
    S_test = build_satisfiability_matrix(X_test, antes)

    train_data = {
        'x': X_train,
        'y': Y_train,
        'c': C_train,
        'S': S_train,
        'y2': Y2_train,
    }
    valid_data = {
        'x': X_valid,
        'y': Y_valid,
        'c': C_valid,
        'S': S_valid,
        'y2': Y2_valid,
    }
    test_data = {
        'x': X_test,
        'y': Y_test,
        'c': C_test,
        'S': S_test,
        'y2': Y2_test,
    }

    return {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'antes': antes,
        'order': order,
    }
