import argparse
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Dirichlet
import pandas as pd

# from utils import get_freq_itemsets
# from load_data import load_support2, Support2
from load_data import load_data
from utils import build_satisfiability_matrix

def make_support2_encoder(encoder_args):
    n_features = encoder_args['n_features']
    n_hidden = encoder_args['n_hidden']
    encoding_dim = encoder_args['encoding_dim']

    return nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.ReLU(True),
        nn.Linear(n_hidden, encoding_dim)
    )

class CEN_BRL(nn.Module):
    def __init__(self, encoder_args, encoding_dim, generator_hidden,
                attention_hidden, A, max_len, n_train):
        super(CEN_BRL, self).__init__()

        self.A = A
        self.n_antes = len(A)

        self.context_encoder = make_support2_encoder(encoder_args)

        self.list_generator = nn.LSTM(
            input_size=self.n_antes,
            hidden_size=encoding_dim,
            num_layers=1
        )

        self.max_len = max_len

        self.attention = nn.Sequential(
            nn.Linear(n_train + encoding_dim, attention_hidden),
            nn.ReLU(True),
            nn.Linear(attention_hidden, 1),
            nn.LogSoftmax(dim=0)
        )

    def forward(self, context, S):
        """
        context - batch_size x n_features
        S - batch_size x n_antecedents

        For differentiability, at each timestep, output a soft weighting over all
        antecedents
        """
        n_train, n_antes = S.size()

        phi = self.context_encoder(context)
        phi = phi.unsqueeze(0)

        generator_state = phi, torch.zeros_like(phi)
        x = torch.ones(1, n_train, self.n_antes)

        d = []
        for t in range(self.max_len):
            e, generator_state = self.list_generator(x, generator_state)

            e_t = e.mean(1)
            attention_input = torch.cat([S.transpose(0, 1), e_t.repeat(n_antes, 1)], dim=-1)

            attention_scores = self.attention(attention_input)

            d.append(attention_scores)

            x = S[:, attention_scores.argmax()].unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.n_antes)

        d = torch.cat(d, dim=-1).transpose(0, 1)

        return d

class CEN_BRL_v2(nn.Module):
    def __init__(self, encoder_args, encoding_dim, generator_hidden,
                attention_hidden, A, max_len, n_train):
        super(CEN_BRL_v2, self).__init__()

        self.A = A
        self.n_antes = len(A)

        self.context_encoder = make_support2_encoder(encoder_args)

        self.list_generator = nn.LSTM(
            input_size=self.n_antes,
            hidden_size=encoding_dim,
            num_layers=1
        )

        self.max_len = max_len

        self.attention = nn.Sequential(
            nn.Linear(n_train + encoding_dim, attention_hidden),
            nn.ReLU(True),
            nn.Linear(attention_hidden, 1),
            nn.LogSoftmax(dim=0)
        )

        self.ante_encoder = nn.Linear(n_train, attention_hidden)
        self.et_encoder = nn.Linear(encoding_dim, attention_hidden)
        self.score = nn.Sequential(
            nn.Linear(attention_hidden, 1),
            nn.LogSoftmax(dim=0)
        )

    def forward(self, context, x):
        """
        context - batch_size x n_features
        S - batch_size x n_antecedents

        For differentiability, at each timestep, output a soft weighting over all
        antecedents
        """
        S = torch.tensor(build_satisfiability_matrix(x, self.A), dtype=torch.float)
        # print(S.size())

        n_train, n_antes = S.size()

        phi = self.context_encoder(context)
        phi = phi.unsqueeze(0)

        generator_state = phi, torch.zeros_like(phi)
        x_t = torch.ones(1, n_train, self.n_antes)

        ante_encoded = self.ante_encoder(S.transpose(0, 1))

        d_soft = []
        d = []
        # available = torch.ones(n_antes, dtype=torch.uint8)

        for t in range(self.max_len):
            e, generator_state = self.list_generator(x_t, generator_state)

            e_t = e.mean(1)
            et_encoded = self.et_encoder(e_t)

            u = self.score(torch.tanh(et_encoded + ante_encoded)).squeeze()
            # up = torch.where(available, u, torch.tensor(np.NINF))
            up = u

            # available[up.argmax()] = 0
            d.append(int(up.argmax()))
            # d.append(self.A[up.argmax()])
            d_soft.append(u)

            S = torch.tensor(build_satisfiability_matrix(x, self.A, prefix=d),
                            dtype=torch.float)
            S = S.unsqueeze(0)

            # print(S.size())

            x_t = u.unsqueeze(0).unsqueeze(0) * S
            # x_t = u.transpose(0, 1).unsqueeze(1) * S

        d_soft = torch.stack(d_soft)
        # d_soft = torch.cat(d_soft, dim=-1).transpose(0, 1)

        return d, d_soft


def create_model(args, antecedents, n_train):
    # TODO: model hyperparameters should be arguments

    encoding_dim = 100

    encoder_args = {
        'n_features': args['n_features'],
        'n_hidden': 100,
        'encoding_dim': encoding_dim
    }

    # model = CEN_BRL(encoder_args, encoding_dim, 200, 100, antecedents, 5, n_train)
    model = CEN_BRL_v2(encoder_args, encoding_dim, 200, 100, antecedents, 3, n_train)

    return model

def compute_B_slow(d, x, y):
    n_train, n_classes = y.shape
    classes = y.argmax(1)

    B = torch.zeros((n_train, n_classes, len(d) + 1))
    for i, xi in enumerate(x):
        for j, lhs in enumerate(d):
            if set(lhs).issubset(xi):
                B[i, classes[i], j] = 1.
                break

        B[i, classes[i], -1] = 1 - B[i, classes[i], :-1].sum()

    # assert B.sum() == S.size()[0]

    return B

def compute_B(d, x, y, S):
    """
    S[i,j] = 1 if datapoint x_i satisfies antecendent j, 0 otherwise

    B[i,k] = 1 if datapoint x_i satisfies d[k], but not d[0], ..., d[k-1]
    """

    n_train, n_classes = y.shape
    _, n_antes = S.shape
    classes = y.argmax(1)

    B = torch.zeros((n_train, n_classes, len(d) + 1))

    unsatisfied = torch.ones(n_train)

    for j, a_idx in enumerate(d):
        new_sat = S[:, a_idx] * unsatisfied

        # mask datapoints that have been satisfied by the ith antecedent
        # so that the unmasked datapoints have not been satisfied yet
        unsatisfied -= new_sat

        B[torch.arange(n_train), classes, j] = new_sat

    # if no rules have been satisfied so far, the null rule must satisfy it,
    # which is the same as saying at the sum for each datapoint is 1.
    B[torch.arange(n_train), classes, -1] = 1 - B[torch.arange(n_train), classes, :-1].sum(-1)

    return B

def get_prior(B, d_len, n_classes, alpha=1.):
    """
    d_len should include the null rule
    """
    priors = alpha + B.sum(0)

    thetas = torch.zeros((d_len, n_classes))
    for i in range(d_len):
        p_theta = Dirichlet(torch.tensor(priors[:, i]))
        thetas[i] = p_theta.rsample()

    return thetas

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_file')
    parser.add_argument('categorical_file')
    parser.add_argument('label_file')

    return vars(parser.parse_args())

def main():
    args = parse_arguments()

    x, c, y, S, antes = load_data(args, 'support2')

    print(x[:5])
    sys.exit()

    print(antes[:5])
    print(type(antes))
    print(len(antes))

    # create model
    args['n_features'] = c.shape[-1]

    model = create_model(args, antes, S.shape[0])

    # use whole dataset for now
    S = torch.tensor(S, dtype=torch.float)
    c = torch.tensor(c, dtype=torch.float)
    n_train, n_classes = y.shape
    classes = y.argmax(1)

    print(f"# datapoints: {n_train}, # classes: {n_classes}")

    # optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for ep in range(50):
        optimizer.zero_grad()

        d, d_soft = model(c, x)
        d_antes = [antes[idx] for idx in d]

        maxes, argmaxes = d_soft.max(dim=-1)

        print(d)
        print(d_antes)

        partial_sum = maxes[:-1].sum()

        sum_log_p_yx = torch.zeros(1, dtype=torch.float)

        B = compute_B(d, x, y, S)

        # assumes decision list length is at least 2
        unsat = torch.ones(n_train) - B[:, :, :-2].sum(-1).sum(-1)
        # print(B[:, :, :-2].sum(-1).sum(-1).size())

        for k, ante in random.sample(list(enumerate(antes)), 80):
            B[torch.arange(n_train), classes, -2] = unsat * S[:, k]
            B[torch.arange(n_train), classes, -1] = 1 - B[torch.arange(n_train), classes, :-1].sum(-1)

            thetas = get_prior(B, len(d) + 1, y.shape[-1], alpha=1.)

            log_py = 0
            for j in range(len(d) + 1):
                # print((B[torch.arange(n_train), classes, j] * torch.log(thetas[j, classes])).size())
                log_py += (B[torch.arange(n_train), classes, j] * torch.log(thetas[j, classes])).sum()

            log_pd = partial_sum + d_soft[-1, k]

            sum_log_p_yx += log_pd + log_py

        (-sum_log_p_yx).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        elapsed = time.time() - start_time
        log_prob = float(sum_log_p_yx)
        print(f"Epoch {ep}: log-prob: {log_prob:.2f} (Elapsed: {elapsed:.2f}s)")

if __name__ == '__main__':
    main()