import argparse
import sys
import time

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

def create_model(args, antecedents, n_train):
    # TODO: model hyperparameters should be arguments

    encoding_dim = 100

    encoder_args = {
        'n_features': args['n_features'],
        'n_hidden': 100,
        'encoding_dim': encoding_dim
    }

    model = CEN_BRL(encoder_args, encoding_dim, 200, 100, antecedents, 5, n_train)

    return model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_file')
    parser.add_argument('categorical_file')
    parser.add_argument('label_file')

    return vars(parser.parse_args())

def main():
    args = parse_arguments()

    x, c, y, S, antes = load_data(args, 'support2')

    print(antes[:5])
    print(type(antes))
    print(len(antes))

    # create model
    args['n_features'] = c.shape[-1]

    model = create_model(args, antes, S.shape[0])

    # use whole dataset for now
    S = torch.tensor(S, dtype=torch.float)
    c = torch.tensor(c, dtype=torch.float)
    n_train = S.shape[0]
    print(n_train)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for ep in range(50):
        optimizer.zero_grad()

        d_soft = model(c, S)

        # print(d)
        maxes, argmaxes = d_soft.max(dim=-1)

        d = [antes[amax] for amax in argmaxes]
        print(d)

        n_classes = y.shape[-1]
        B = torch.zeros((n_train, n_classes, len(d) + 1))
        for i, xi in enumerate(x):
            for j, lhs in enumerate(d):
                if set(lhs).issubset(xi):
                    B[i, y[i, 0], j] = 1.
                    break

            B[i, y[i, 0], -1] = 1 - B[i, y[i, 0], :-1].sum()

        assert B.sum() == S.size()[0]

        # get dirichlet prior
        alpha = 1.
        priors = alpha + B.sum(0)

        thetas = torch.zeros((len(d) + 1, n_classes))
        for i in range(len(d) + 1):
            p_theta = Dirichlet(torch.tensor(priors[:, i]))
            thetas[i] = p_theta.rsample()

        # compute p(y | d)
        log_py = 0
        for i, yi in enumerate(y):
            for j in range(len(d) + 1):
                log_py += B[i, y[i, 0], j] * torch.log(thetas[j, y[i, 0]])

        # compute p(d | input), as p(d_1 | input) p(d_2 | d_1, input) ...
        log_pd = maxes.sum()

        log_prob = -(log_py + log_pd)

        elapsed = time.time() - start_time
        print(f"Epoch {ep}: log-prob: {log_prob:.2f}, log p(d|x): {log_pd:.2f}, ", end='')
        print(f"log p(y|d): {log_py:.2f} (Elapsed: {elapsed:.2f}s)")

        log_prob.backward()
        optimizer.step()

if __name__ == '__main__':
    main()