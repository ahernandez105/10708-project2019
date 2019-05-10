import time
import argparse
import sys
import pprint
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score

from cen_brl import make_support2_encoder
from load_data.load_data import load_data
from load_data.support2 import Support2
from load_data.imdb import IMDB
from load_data.mnist import MNIST
from models import CEN_RCN, FF, CEN_RCN_Simple
from metrics import accuracy_score_self, rae_loss_self

def rae(y_true, y_pred):
    n_classes = y_true.shape[-1]

    expected_death = (torch.arange(n_classes, dtype=torch.float) * py).sum(-1)
    diff = expected_death - batch_classes.type_as(expected_death)
    rae = torch.min(torch.tensor(1.), torch.abs(diff / expected_death))

class IMDBEncoder(nn.Module):
    def __init__(self, encoder_args):
        super(IMDBEncoder, self).__init__()

        self.vocab_size = encoder_args['vocab_size']
        self.n_hidden = encoder_args['n_hidden']
        self.encoding_dim = encoder_args['encoding_dim']

        print(f"# vocab: {self.vocab_size}")

        self.embedding = nn.Embedding(self.vocab_size, self.n_hidden)
        self.encoder = nn.LSTM(input_size=self.n_hidden, hidden_size=self.encoding_dim,
                            batch_first=False)

    def forward(self, x):
        embeddings = self.embedding(x)
        out, _ = self.encoder(embeddings)

        # take the last timestep
        return out[:, -1]

class ImageEncoder(nn.Module):
    def __init__(self, encoder_args):
        super(ImageEncoder, self).__init__()

        self.n_channels = encoder_args['n_channels']
        self.encoding_dim = encoder_args['encoding_dim']

        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 1, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=3),
            nn.MaxPool2d(2, padding=1),
            nn.ReLU(True)
        )

        # final is 6x6
        self.final = nn.Linear(6 * 6, self.encoding_dim)

    def forward(self, x):
        h = self.encoder(x)

        # flatten
        return self.final(h.flatten(start_dim=1, end_dim=-1))

def create_model(args, pyz):
    encoding_dim = args['encoding_dim']

    encoder_args = {
        'n_hidden': args['n_encoder_hidden'],
        'encoding_dim': encoding_dim
    }
    if args['dataset'] == 'support2':
        n_features = args['n_features']
        encoder_args['n_features'] = n_features
        context_encoder = make_support2_encoder(encoder_args)
    elif args['dataset'] == 'imdb':
        # encoder_args['vocab_size'] = 88587 + 3 # hardcode for now
        encoder_args['vocab_size'] = args['vocab_size']
        context_encoder = IMDBEncoder(encoder_args)
    elif args['dataset'] == 'mnist':
        encoder_args['n_channels'] = 1
        context_encoder = ImageEncoder(encoder_args)

    # for now, x = c
    n_hidden = args['n_rcn_hidden']

    if args['model'] == 'rcn':
        n_features = args['n_features']
        model = CEN_RCN(context_encoder, encoding_dim, n_features, n_hidden,
                        pyz,
                        mask_u=args['rcn_mask'],
                        pyz_learnable=args['pyz_learnable'])

    elif args['model'] == 'simple_rcn':
        model = CEN_RCN_Simple(context_encoder, encoding_dim,
                            pyz,
                            mask_u=args['rcn_mask'],
                            pyz_learnable=args['pyz_learnable'])

    elif args['model'] == 'ff':
        model = FF(context_encoder, encoding_dim, args['n_classes'])

    return model

def eval_support2(y_true, y_pred):
    """
    Calculate RAE and acc at percentiles
    """

    percentile_cutoffs = [1, 7, 32]

    # compute percentile ground truths and predictions
    # for percentile in [25, 50, 75]:
    for percentile, cutoff in zip([25, 50, 75], percentile_cutoffs):
        # percentile_cutoff = np.percentile(y_true, percentile)

        y_true_p = np.where(y_true <= cutoff, 1, 0)
        y_pred_p = np.where(y_pred <= cutoff, 1, 0)

        acc_score = accuracy_score(y_true_p, y_pred_p) * 100

        print(f"{percentile}th percentile: cutoff {cutoff} - acc {acc_score:.2f}%")

def compute_pyz(n_train, n_classes, n_antes, Y, S, alpha):
    # build count matrix
    # counts[y, i]: number of datapoints with label y that satisfy
    #   antecedent i
    big_counts = np.zeros((n_train, n_classes, n_antes), dtype=np.uint8)
    big_counts[np.arange(n_train), Y] = S
    counts = big_counts.sum(0).astype(np.float)

    assert (n_classes, n_antes) == counts.shape
    assert np.allclose(counts.sum(0), S.sum(0))

    counts += alpha

    return (counts / counts.sum(0)).transpose()

def train_model(args, model, dataloader, valid_dataloader=None):
    start_time = time.time()
    n_epochs = args['n_epochs']
    device = args['device']

    # build optimizers

    if args['optim'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    elif args['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    else:
        raise NotImplementedError

    # early stopping metrics
    best_val_acc = 0.
    no_improvement_vals = 0
    best_model = None

    for ep in range(1, n_epochs + 1):
        total_log_prob = 0.
        for batch_sample in dataloader:
            
            batch_c = batch_sample['context'].to(device)
            batch_s = batch_sample['S'].to(device)
            batch_classes = batch_sample['y'].argmax(1).to(device)
            batch_len = len(batch_classes)

            optimizer.zero_grad()

            if args['model'] == 'rcn':
                pz, py = model(batch_c, batch_c, batch_s)
                # compute p(y | x) = \sum_z p(y|z) p(z|x)
                # p(z|x) is model output
                # p(y|z) is precomputed
                # py = torch.matmul(pz, pyz)

            elif args['model'] == 'simple_rcn':
                pz, py = model(batch_c, batch_s)

            elif args['model'] == 'ff':
                py = model(batch_c)

            # avoid nan when py = 0
            py = py + 1e-12
            log_prob = py.log()[torch.arange(batch_len), batch_classes].sum()
            # log_prob = py[torch.arange(batch_len, device=device), batch_classes].sum()

            # compute/minimize RAE?
            # also other metrics
            # RAE = min(1, |(p-t) / p|),
            # where p and t are the predicted and true survival times
            # expected_death = (torch.arange(n_classes, dtype=torch.float) * py).sum(-1)
            # diff = expected_death - batch_classes.type_as(expected_death)
            # rae = torch.min(torch.tensor(1.), torch.abs(diff / expected_death))
            # rae.mean().backward()

            if not torch.isfinite(log_prob):
                print("log_prob not finite??")
                raise RuntimeError
            total_log_prob += float(log_prob)

            (-log_prob).backward()

            optimizer.step()

        if ep % 5 == 0 and valid_dataloader is not None:
            # validation

            all_preds = []
            all_py = []
            valid_y = []
            with torch.no_grad():
                for batch_sample in valid_dataloader:
                    batch_c = batch_sample['context'].to(device)
                    batch_s = batch_sample['S'].to(device)

                    if args['model'] == 'rcn':
                        pz, py = model(batch_c, batch_c, batch_s)
                        # py = torch.matmul(pz, pyz)

                    elif args['model'] == 'simple_rcn':
                        pz, py = model(batch_c, batch_s)
                    
                    elif args['model'] == 'ff':
                        py = model(batch_c)

                    predictions = py.argmax(-1)
                    all_preds.append(predictions)
                    all_py.append(py)
                    valid_y.append(batch_sample['y'].numpy())

            all_preds = torch.cat(all_preds)
            all_py = torch.cat(all_py)
            valid_y = np.concatenate(valid_y).argmax(-1)
            valid_acc = accuracy_score(valid_y, all_preds.cpu().numpy())

            print(all_preds[:10])
            print(valid_y[:10])
            # print(diff[:10])

            log_line = f"Ep {ep:<5} -  log prob: {total_log_prob:.4f}, validation acc: {valid_acc:.4f}"

            # RAE
            if args['dataset'] == 'support2':
                n_classes = all_py.size()[-1]
                expected_death = (np.arange(n_classes) * all_py.cpu().numpy()).sum(-1)
                diff = expected_death - valid_y
                valid_rae = np.minimum(1, np.abs(diff)).mean()

                log_line += f", validation RAE: {valid_rae:.4f}"

            log_line += "\t({:.3f}s)".format(time.time() - start_time)
            print(log_line)

            # early stopping
            if valid_acc >= best_val_acc:
                best_val_acc = valid_acc
                no_improvement_vals = 0
                # save best model so far
                best_model = model.state_dict()

            else:
                no_improvement_vals += 1
                if no_improvement_vals > args['patience']:
                    print("Early stopping!")
                    break

        if not torch.isfinite(log_prob).all():
            print("Error!")
            print(py)
            print(py.min())
            print(py.log().min())
            print(py.log().max())
            raise RuntimeError

    # load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model

def pred_model(args, model, dataloader):
    device = args['device']

    # make predictions with associated antecedents
    all_preds = []
    all_py = []
    all_pz = []
    with torch.no_grad():
        for batch_sample in dataloader:
            batch_c = batch_sample['context'].to(device)
            batch_s = batch_sample['S'].to(device)

            if args['model'] == 'rcn':
                pz, py = model(batch_c, batch_c, batch_s)
                # py = torch.matmul(pz, pyz)

                all_pz.append(pz)

            elif args['model'] == 'simple_rcn':
                pz, py = model(batch_c, batch_s)

                all_pz.append(pz)
            
            elif args['model'] == 'ff':
                py = model(batch_c)

            predictions = py.argmax(-1)

            all_py.append(py)
            all_preds.append(predictions)

    all_preds = torch.cat(all_preds)
    all_py = torch.cat(all_py, dim=0)
    all_pz = torch.cat(all_pz, dim=0)

    return all_preds, all_py, all_pz

def evaluate_model(model, dataloader):
    all_preds = []
    for batch_sample in dataloader:
        batch_c = batch_sample['context'].to(device)
        batch_s = batch_sample['S'].to(device)

        if args['model'] == 'rcn':
            pz, py = model(batch_c, batch_c, batch_s)
            # py = torch.matmul(pz, pyz)
        
        elif args['model'] == 'ff':
            py = model(batch_c)

        predictions = py.argmax(-1)
        all_preds.append(predictions)

    all_preds = torch.cat(all_preds)
    valid_acc = accuracy_score(valid_classes, all_preds)

def compute_support2_metrics(args, y_pred, y_probs, y_true, y_true_full):
    """
    Compute RAE and Acc@X metrics on predictions
    """

    test_classes = y_true.argmax(-1)

    # print(all_py[:5])
    print(y_pred[:10])
    print(y_true[:10])

    # get distribution of predictions
    pred_y = np.zeros_like(y_true)
    pred_y[np.arange(pred_y.shape[0]), y_probs.argmax(-1)] = 1
    print("Pred distribution:", pred_y.sum(0).astype(int).tolist())

    print("Test distribution:", y_true.sum(0).astype(int).tolist())
    print("Test accuracy: {:.4f}".format(accuracy_score(test_classes, y_pred)))

    # get Acc@T and RAE metrics on test set
    accs = {}
    for t in [1, 7, 32]:
        accs[t] = accuracy_score_self(y_true_full, y_probs, t)

    n_classes = y_probs.shape[-1]
    expected_death = (np.arange(n_classes) * y_probs).sum(-1)
    diff = expected_death - test_classes
    test_rae = np.minimum(1, np.abs(diff)).mean()
    print(f"Test RAE: {test_rae:.4f}")

    rae_nc = rae_loss_self(y_true_full, y_probs)

    return accs, test_rae, rae_nc

def write_results(results, outfile):
    f = open(outfile, 'w')
    writer = csv.DictWriter(f, fieldnames=results.keys())

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['rcn', 'ff', 'simple_rcn'])
    parser.add_argument('dataset', choices=['support2', 'imdb', 'mnist'])

    # support2 args
    parser.add_argument('--raw_file')
    parser.add_argument('--categorical_file')
    parser.add_argument('--label_file')
    parser.add_argument('--use_missing', action='store_true')

    # imdb args
    parser.add_argument('--vocab_file')

    # mnist args
    parser.add_argument('--interp_file')
    parser.add_argument('--interp_type',
                        choices=['pixels16x16', 'pixels7x7'],
                        default='pixels7x7'
                        )

    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--csv')
    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=200)

    parser.add_argument('--print_rules', action='store_true')
    parser.add_argument('--outfile')

    # model hyperparameters
    parser.add_argument('--encoding_dim', type=int, default=150)
    parser.add_argument('--n_encoder_hidden', type=int, default=150)
    parser.add_argument('--n_rcn_hidden', type=int, default=150)
    parser.add_argument('--rcn_mask', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_support', type=int, default=30)
    parser.add_argument('--max_lhs', type=int, default=3)
    parser.add_argument('--loss', choices=['log_prob', 'rae'], default='log_prob')
    parser.add_argument('--pyz_learnable', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--max_vocab', type=int)

    parser.add_argument('--optim', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=0.001)

    args = vars(parser.parse_args())

    if args['dataset'] == 'support2':
        assert args['raw_file'] is not None
        # assert args['categorical_file'] is not None
        assert args['label_file'] is not None
    elif args['dataset'] == 'imdb':
        assert args['raw_file'] is not None
        assert args['vocab_file'] is not None
    elif args['dataset'] == 'mnist':
        assert args['raw_file'] is not None
        assert args['interp_file'] is not None

    return args

def main():
    """
    TODO:
    - subsample antecedents
    - add support for MNIST and IMDB
    """
    args = parse_arguments()

    # train_data, valid_data, test_data, antes = load_data(args, args['dataset'])
    data = load_data(args, args['dataset'])

    train_data = data['train_data']
    valid_data = data['valid_data']
    test_data = data['test_data']
    antes = data['antes']

    if args['dataset'] == 'support2':
        args['n_features'] = train_data['c'].shape[-1]
    elif args['dataset'] == 'imdb':
        args['vocab_size'] = data['vocab_size']

    for d in [train_data, valid_data, test_data]:
        for k, v in d.items():
            if type(v) is list:
                print(k, len(v))
            else:
                print(k, v.shape)

        print('---')

    print(train_data['c'].shape)

    if args['cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    args['device'] = device

    n_antes = len(antes)
    print(train_data['S'].shape)

    n_train, n_classes = train_data['y'].shape
    args['n_classes'] = n_classes
    classes = train_data['y'].argmax(1)
    print("Class distribution:", train_data['y'].sum(0).astype(int).tolist())

    # sanity check - classification acc if just predict majority class
    majority_class = train_data['y'].sum(0).argmax()
    valid_classes = valid_data['y'].argmax(-1)
    majority_valid = accuracy_score(valid_classes, np.full_like(valid_classes, majority_class))
    print(f"Majority acc on validation: {majority_valid:.4f}")

    # p(y|z): normalized counts of number of classes that satisfy each antecedent
    pyz = compute_pyz(n_train, n_classes, n_antes, classes, train_data['S'], args['alpha'])
    pyz = torch.tensor(pyz, dtype=torch.float).to(device)

    # create model
    model = create_model(args, pyz).to(device)

    print("Params:")
    for n, p in model.named_parameters():
        print(n, p.size(), p.requires_grad)
    print("---")

    batch_size = args['batch_size']

    if args['dataset'] == 'support2':
        dataset_class = Support2
    
    elif args['dataset'] == 'imdb':
        dataset_class = IMDB

    elif args['dataset'] == 'mnist':
        dataset_class = MNIST

    else:
        raise NotImplementedError

    train_dataset = dataset_class(train_data['x'], train_data['c'], train_data['y'], train_data['S'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    valid_dataset = dataset_class(valid_data['x'], valid_data['c'], valid_data['y'], valid_data['S'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_dataset = dataset_class(test_data['x'], test_data['c'], test_data['y'], test_data['S'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = train_model(args,
                        model,
                        train_loader,
                        valid_dataloader=valid_loader)

    all_preds, all_py, all_pz = pred_model(args, model, test_loader)
    print(all_preds[:5])

    all_preds = all_preds.cpu().numpy()
    all_py = all_py.cpu().numpy()
    all_pz = all_pz.cpu().numpy()

    if args['dataset'] == 'support2':
        accs, test_rae, rae_nc = compute_support2_metrics(args,
                                                        all_preds,
                                                        all_py,
                                                        test_data['y'],
                                                        test_data['y2'])

    test_classes = test_data['y'].argmax(-1)

    if args['model'] in ['rcn', 'simple_rcn'] and args['print_rules']:
        # all_pz = torch.cat(all_pz, dim=0)
        selected_antes = [antes[i] for i in all_pz.argmax(-1)]

        for i, (a, idx) in enumerate(zip(selected_antes, all_pz.argmax(-1))):
            print(a, pyz[idx])
            print(pyz[idx, test_classes[i]])
            if i >= 4:
                break

    if args['dataset'] == 'support2':
        eval_support2(test_classes, all_preds)

        summary = {
            'encoding_dim': args['encoding_dim'],
            'n_encoder_hidden': args['n_encoder_hidden'],
            'n_rcn_hidden': args['n_rcn_hidden'],
            'min_support': args['min_support'],
            'max_lhs': args['max_lhs'],
            'n_antes': n_antes,
            'rae': rae_nc,
        }
        for t, acc in accs.items():
            summary[f'Acc@{t}'] = acc
        
        headers = [
            'encoding_dim',
            'n_encoder_hidden',
            'n_rcn_hidden',
            'min_support',
            'max_lhs',
            'n_antes',
            'rae',
            'Acc@1',
            'Acc@7',
            'Acc@32'
        ]

        line = ','.join([str(summary[h]) for h in headers])
        print(line)

    elif args['dataset'] == 'imdb':
        acc_score = accuracy_score(test_classes, all_preds)
        print(f"Accuracy on test: {acc_score:.7f}")

    elif args['dataset'] == 'mnist':
        acc_score = accuracy_score(test_classes, all_preds)
        print(f"Accuracy on test: {acc_score:.7f}")

    else:
        raise NotImplementedError

    if args['outfile']:
        with open(args['outfile'], 'a') as f:
            f.write(line)
            f.write('\n')

if __name__ == '__main__':
    main()