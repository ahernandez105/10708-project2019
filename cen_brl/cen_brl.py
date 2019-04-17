import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Dirichlet
import pandas as pd

from utils import get_freq_itemsets
from load_data import load_support2, Support2

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
        """
        A - list of antecedents
        """

        super(CEN_BRL, self).__init__()

        self.context_encoder = make_support2_encoder(encoder_args)

        # for now, the decision list generator takes in 
        # the encoded context (phi) at each timestep
        self.list_generator = nn.LSTM(
            input_size=encoding_dim,
            hidden_size=generator_hidden,
            num_layers=1
        )

        # max length of decision list
        self.max_len = max_len

        self.A = A

        print("# train", n_train)
        self.attention = nn.Sequential(
            nn.Linear(n_train + generator_hidden, attention_hidden),
            nn.ReLU(True),
            nn.Linear(attention_hidden, 1),
        )

    def forward(self, context, S):
        """
        context - batch_size x n_features
        S - batch_size x n_antecedents
        """
        n_train, n_antes = S.size()
        
        phi = self.context_encoder(context)
        phi = phi.unsqueeze(0)

        d = []

        for t in range(self.max_len):
            if t == 0:
                e, generator_state = self.list_generator(phi)
            else:
                e, generator_state = self.list_generator(phi, generator_state)

            # compute e_t by averaging...
            # e_t - 1 x n_hidden
            e_t = e.mean(1)

            # want scores for each ante,
            # so input is n_ante x (n_train + gen_features)
            attention_input = torch.cat([S.transpose(0, 1), e_t.repeat(n_antes, 1)], dim=-1)

            attention_scores = self.attention(attention_input)

            a_max = attention_scores.argmax()
            d.append(self.A[a_max])

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

    # load data
    x, c, y = load_support2(args['raw_file'], args['categorical_file'],
                    args['label_file'])
    S, ante_lens, antes = get_freq_itemsets(x, y)

    dataset = Support2(x, c, y, S)

    print(antes[:5])

    # create model
    args['n_features'] = c.shape[-1]

    model = create_model(args, antes, S.shape[0])

    # use whole dataset for now
    S = torch.tensor(S, dtype=torch.float)
    c = torch.tensor(c, dtype=torch.float)

    for ep in range(50):
        d = model(c, S)

        n_classes = y.shape[-1]
        print(n_classes)
        B = torch.zeros((len(dataset), n_classes, len(d) + 1))
        print(y.shape)
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
        print(priors)

        thetas = torch.zeros((len(d) + 1, n_classes))
        for i in range(len(d) + 1):
            p_theta = Dirichlet(torch.tensor(priors[:, i]))
            thetas[i] = p_theta.rsample()

        print(thetas)

        # compute p(y | d)
        log_prob = 0
        for i, yi in enumerate(y):
            for j in range(len(d) + 1):
                log_prob += B[i, y[i, 0], j] * torch.log(thetas[j, y[i, 0]])

        log_prob.backward()
        sys.exit()

if __name__ == '__main__':
    main()