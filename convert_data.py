"""
Convert data into format for BRL_code

.tab - inputs, just the values
.Y - class labels

replace missing data with 'missing'
or impute with suggested values?

we can only work with categorical features, so split into intervals

take d.time as time of death?????

TODO:
- add ca to ca features (for ca_yes, ca_no)
- add feature name to missing? (e.g. adlp_missing)
- or remove missing values?
"""

import sys
import csv
import pprint
from collections import OrderedDict

import numpy as np

support2_fillin = {
    'alb': 3.5,
    'pafi': 333.3,
    'bili': 1.01,
    'crea': 1.01,
    'bun': 6.51,
    'wblc': 9,
    'urine': 2502,
}

continuous_vars = [
    'age',
    'slos',
    'd.time',
    'num.co',
    'edu',
    'scoma', # not really?
    'charges',
    'totcst',
    'totmcst',
    'avtisst',
    'sps',
    'aps',
    'surv2m',
    'surv6m',
    'hday',
    'prg2m',
    'prg6m',
    'dnrday',
    'meanbp',
    'wblc',
    'hrt',
    'resp',
    'temp',
    'pafi',
    'alb',
    'bili',
    'crea',
    'sod',
    'ph',
    'adlp',  # maybe not? activities of daily living
    'adls', # adl surrogate
    'adlsc',
]

mapping = {
    'age': float,
    'death': bool,
    'sex': str,
    'hospdead': bool,
    'slos': int,
    'd.time': int,
    'dzgroup': str,
    'dzclass': str,
    'num.co': int,
    'edu': int,
    'income': str,
    'scoma': int,
    'charges': float,
    'totcst': float,
    'totmcst': float,
    'avtisst': float,
    'race': str,
    'sps': float,
    'aps': int,
    'surv2m': float,
    'surv6m': float,
    'hday': int,
    'diabetes': bool,
    'dementia': bool,
    'ca': str,
    'prg2m': float,
    'prg6m': float,
    'dnr': str,
    'dnrday': int,
    'meanbp': float,
    'wblc': float,
    'hrt': float,
    'resp': int,
    'temp': float,
    'pafi': float,
    'alb': float,
    'bili': float,
    'crea': float,
    'sod': int,
    'ph': float,
    'glucose': float,
    'bun': float,
    'urine': float,
    'adlp': int,
    'adls': int,
    'sfdm2': str,
    'adlsc': float,
}

def convert_support2(infile, outfile, targetfile, target_name):
    f = open(infile, 'r')
    reader = csv.DictReader(f)
    # pprint.pprint(reader.fieldnames)
    # sys.exit()
    keys = reader.fieldnames[1:]

    # consider up to 156 weeks
    d_max = 1092

    # collect all values
    all_data = []
    for line in reader:
        # ignore index
        items = list(line.items())[1:]

        # map to appropriate type
        mapped_items = OrderedDict.fromkeys(keys)
        for k, v in items:
            if v == '':
                pass
            else:
                try:
                    if mapping[k] is bool:
                        v = bool(int(v.strip()))
                        v = f'{k}_{v}'
                    else:
                        v = mapping[k](v.strip())
                except:
                    print(f"[{k}] [{v.strip()}]")
                    sys.exit()

                # mapped_items[k] = mapping[k](v)
                if mapping[k] == str:
                    v = v.replace(' ', '_')
                    # mapped_items[k] = mapped_items[k].replace(' ', '_')

                mapped_items[k] = v

        # mapped_items = OrderedDict(mapped_items)
        all_data.append(mapped_items)

    f.close()

    binned_data = [OrderedDict.fromkeys(keys) for _ in all_data]

    # get ranges of int and float values
    for k, t in mapping.items():
        if t is float or t is int and k != 'd.time':
            min_val = None
            max_val = None
            values = set()
            for row in all_data:
                if row[k] is not None:
                    if min_val is None or row[k] < min_val:
                        min_val = row[k]
                    if max_val is None or row[k] > max_val:
                        max_val = row[k]
                    values.add(row[k])

            n_distinct = len(values)
            print(f"{k:10} - Max: {max_val:<12} Min: {min_val:<12} # distinct values: {n_distinct}")

            # split evenly into 10 bins. if n_distince is at most 10, then just use those values
            if n_distinct > 10:
                bins = np.linspace(min_val, max_val, 11)

                # name of feature is {min} < {feat} < {max}
                feats = {}
                for i, lower in enumerate(bins[:-1]):
                    upper = bins[i+1]
                    feats[(lower, upper)] = f"{lower:.2f}<{k}<{upper:.2f}" 
                
                for old, new in zip(all_data, binned_data):
                    if old[k] is not None:
                        for (lower, upper), feat_name in feats.items():
                            if old[k] >= lower and old[k] <= upper:
                                new[k] = feat_name
                                break

                        if new[k] is None:
                            print(feats)
                            print(old[k])
                            raise ValueError
            else:
                for old, new in zip(all_data, binned_data):
                    if old[k] is not None:
                        new[k] = f'{k}_{old[k]}'

        elif k == 'd.time':
            # split into 156 week intervals
            # bins = np.arange(0, 7*156+1, 7)
            # split into 13 84-day intervals
            bins = np.arange(0, 7*156+1, 84)
            feats = {}
            for i, lower in enumerate(bins[:-1]):
                upper = bins[i+1]
                feats[(lower, upper)] = f"{lower}<{k}<{upper}" 
            
            for old, new in zip(all_data, binned_data):
                if old[k] is None:
                    raise ValueError('d.time is None!')

                for (lower, upper), feat_name in feats.items():
                    if old[k] >= lower and old[k] < upper:
                        new[k] = feat_name
                        break

                if new[k] is None:
                    new[k] = 'd.time>=1092'
        
        else:
            for old, new in zip(all_data, binned_data):
                new[k] = old[k]

    # fill all Nones with missing
    for row in binned_data:
        for k in row.keys():
            if row[k] is None:
                row[k] = 'missing'

    print(binned_data[0])

    # target_name = 'd.time'
    target = [row[target_name] for row in binned_data]

    # remove keys about death?
    # also remove target
    to_remove = ['death', 'hospdead', 'd.time']
    for row in binned_data:
        for k in to_remove:
            del row[k]

    for r in binned_data[:5]:
        print(r)
    print(target[:10])

    # write attributes
    dial = csv.excel_tab
    dial.delimiter = ' '
    dial.lineterminator = '\n'

    for split in ['train', 'valid', 'test']:
        if split == 'train':
            split_data = binned_data[:7105]
            split_target = target[:7105]
        elif split == 'valid':
            split_data = binned_data[7105:8105]
            split_target = target[7105:8105]
        elif split == 'test':
            split_data = binned_data[8105:]
            split_target = target[8105:]

        ofile = os.path.join('data', outfile, outfile + '_' + split + '.tab')
        tfile = os.path.join('data', targetfile, targetfile + '_' + split + '.Y')

        g = open(ofile, 'w')
        # writer = csv.DictWriter(g, split_data[0].keys(), dialect=dial)
        # writer.writerows(split_data)

        # don't write 'missing' values
        for row in split_data:
            values = [str(x) for x in row.values() if x != 'missing']
            g.write(' '.join(values))
            g.write('\n')
        g.close()

        # write target
        g = open(tfile, 'w')

        if target_name == 'd.time':
            feats = []
            # bins = np.arange(0, 7*156+1, 7)
            bins = np.arange(0, 7*156+1, 84)
            for i, lower in enumerate(bins[:-1]):
                upper = bins[i+1]
                feats.append(f"{lower}<d.time<{upper}")
            feats.append('d.time>=1092')

        elif target_name == 'death':
            feats = ['death_True', 'death_False']
        
        else:
            raise NotImplementedError

        for y in split_target:
            # one_hot = [str(int(y == feat)) for feat in feats]
            one_hot = [int(y == feat) for feat in feats]
            try:
                assert sum(one_hot) == 1
            except:
                print(one_hot)
                print(y)
                print(feats)
                raise ValueError
            one_hot = [str(x) for x in one_hot]
            g.write(' '.join(one_hot))
            g.write('\n')
        g.close()


if __name__ == '__main__':
    convert_support2(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])