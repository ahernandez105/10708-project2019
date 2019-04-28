import time
import argparse
import sys
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score

from cen_brl import make_support2_encoder
from load_data import load_data, Support2, load_support2_all, load_data_new
# from support2 import load_data as sup2_load

class CEN_RCN(nn.Module):
    def __init__(self, encoder_args, encoding_dim, n_features, n_hidden):
        super(CEN_RCN, self).__init__()

        self.context_encoder = make_support2_encoder(encoder_args)

        self.fusion = nn.Linear(encoding_dim + n_features, n_hidden)
        self.input_encoder = nn.Linear(n_features, n_hidden)
        
    def forward(self, context, x, S):
        phi = self.context_encoder(context)

        # dot-product attention between rule representations and
        # fused rep of context and attributes

        # input x: n_train x n_features
        # x_rep: n_train x n_hidden
        # S: n_rules x n_train
        # S_rep: n_rules x n_hidden

        # h: n_train x n_hidden
        h = self.fusion(torch.cat([phi, x], dim=-1))

        x_rep = self.input_encoder(x)
        S_rep = torch.matmul(S.transpose(0, 1), x_rep)

        # u: n_train x n_rules
        # scores for each
        u = torch.matmul(h, S_rep.transpose(0, 1))
        pz = F.softmax(u, dim=-1)

        return pz

class FF(nn.Module):
    def __init__(self, encoder_args, encoding_dim):
        super(FF, self).__init__()

        self.context_encoder = make_support2_encoder(encoder_args)

        # self.context_hidden = nn.Linear(encoder_args['n_features'], encoder_args['n_hidden'])
        # self.context_out = nn.Linear(encoder_args['n_hidden'], encoding_dim)

        self.final = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(encoding_dim, 2)
        )

    def forward(self, x):
        # h = F.relu(self.context_hidden(x))
        # phi = self.context_out(h)
        phi = self.context_encoder(x)
        return F.softmax(self.final(phi), dim=-1)

def create_model(args):
    # TODO: model hyperparameters should be arguments

    encoding_dim = 10

    n_features = args['n_features']

    encoder_args = {
        'n_features': n_features,
        'n_hidden': 20,
        'encoding_dim': encoding_dim
    }

    # for now, x = c
    n_hidden = 20

    if args['model'] == 'rcn':
        model = CEN_RCN(encoder_args, encoding_dim, n_features, n_hidden)
    elif args['model'] == 'ff':
        model = FF(encoder_args, encoding_dim)

    return model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['rcn', 'ff'])
    parser.add_argument('raw_file')
    parser.add_argument('categorical_file')
    parser.add_argument('label_file')

    parser.add_argument('--csv')
    parser.add_argument('--seed', type=int, default=42)

    return vars(parser.parse_args())

def main():
    """
    TODO:
    - batching
    - only pick antecedents that satisfy the datapoint
    - also output best antecedent (using argmax)
    """
    args = parse_arguments()

    train_data, valid_data, test_data, antes = load_data_new(args, 'support2')
    for d in [train_data, valid_data, test_data]:
        for k, v in d.items():
            if type(v) is list:
                print(k, len(v))
            else:
                print(k, v.shape)

        print('---')

    print(train_data['c'].shape)

    args['n_features'] = train_data['c'].shape[-1]

    model = create_model(args)

    n_antes = len(antes)
    print(train_data['S'].shape)

    n_train, n_classes = train_data['y'].shape
    classes = train_data['y'].argmax(1)
    print("Class distribution:", train_data['y'].sum(0))

    # build count matrix
    # counts[y, i]: number of datapoints with label y that satisfy
    #   antecedent i
    big_counts = np.zeros((n_train, n_classes, n_antes))
    big_counts[np.arange(n_train), classes] = train_data['S']
    counts = big_counts.sum(0)

    assert (n_classes, n_antes) == counts.shape
    assert np.allclose(counts.sum(0), train_data['S'].sum(0))

    # p(y|z): normalized counts of number of classes that satisfy each antecedent
    pyz = torch.tensor((counts / counts.sum(0)).transpose(), dtype=torch.float)

    print("Params:")
    for n, p in model.named_parameters():
        print(n, p.size(), p.requires_grad)
    print("---")

    batch_size = 64

    train_dataset = Support2(train_data['x'], train_data['c'], train_data['y'], train_data['S'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Support2(valid_data['x'], valid_data['c'], valid_data['y'], valid_data['S'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Support2(test_data['x'], test_data['c'], test_data['y'], test_data['S'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=1)

    start_time = time.time()
    n_epochs = 200
    for ep in range(n_epochs):
        total_log_prob = 0.
        for batch_sample in train_loader:
            
            batch_c = batch_sample['context']
            batch_s = batch_sample['S']
            batch_classes = batch_sample['y'].argmax(1)
            batch_len = len(batch_classes)

            optimizer.zero_grad()

            if args['model'] == 'rcn':
                pz = model(batch_c, batch_c, batch_s)

                # compute p(y | x) = \sum_z p(y|z) p(z|x)
                # p(z|x) is model output
                # p(y|z) is precomputed
                py = torch.matmul(pz, pyz)

            elif args['model'] == 'ff':
                py = model(batch_c)

            # avoid nan when py = 0
            py += 1e-12
            log_prob = py.log()[torch.arange(batch_len), batch_classes].sum()

            if not torch.isfinite(log_prob):
                print("log_prob not finite??")
                raise RuntimeError
            total_log_prob += float(log_prob)

            (-log_prob).backward()

            optimizer.step()

        if ep % 10 == 0:
            # validation

            valid_classes = valid_data['y'].argmax(-1)

            all_preds = []
            with torch.no_grad():
                for batch_sample in valid_loader:
                    batch_c = batch_sample['context']
                    batch_s = batch_sample['S']

                    if args['model'] == 'rcn':
                        pz = model(batch_c, batch_c, batch_s)
                        py = torch.matmul(pz, pyz)
                    
                    elif args['model'] == 'ff':
                        py = model(batch_c)

                    predictions = py.argmax(-1)
                    all_preds.append(predictions)

            all_preds = torch.cat(all_preds)
            valid_acc = accuracy_score(valid_classes, all_preds)

            print(all_preds[:10])
            print(valid_classes[:10])

            log_line = f"Ep {ep:<5} -  log prob: {total_log_prob:.4f}, validation acc: {valid_acc:.4f}"
            log_line += "\t({:.3f}s)".format(time.time() - start_time)
            print(log_line)

        if not torch.isfinite(log_prob).all():
            print("Error!")
            print(py)
            print(py.min())
            print(py.log().min())
            print(py.log().max())
            raise RuntimeError

    # make predictions
    all_preds = []
    all_py = []
    with torch.no_grad():
        for batch_sample in test_loader:
            batch_c = batch_sample['context']
            batch_s = batch_sample['S']

            if args['model'] == 'rcn':
                pz = model(batch_c, batch_c, batch_s)
                py = torch.matmul(pz, pyz)
            
            elif args['model'] == 'ff':
                py = model(batch_c)

            predictions = py.argmax(-1)

            all_py.append(py)
            all_preds.append(predictions)

    all_preds = torch.cat(all_preds)
    all_py = torch.cat(all_py, dim=0)

    test_classes = test_data['y'].argmax(-1)

    print(all_py[:10])
    print(all_preds[:10])
    print(test_classes[:10])
    print("Test distribution:", test_data['y'].sum(0))
    print("Test accuracy: {:.4f}".format(accuracy_score(test_classes, all_preds)))

if __name__ == '__main__':
    main()