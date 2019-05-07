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
from load_data import load_data, Support2
# from support2 import load_data as sup2_load

class CEN_RCN(nn.Module):
    def __init__(self, context_encoder, encoding_dim, n_features, n_hidden, pyz, mask_u=False,
                pyz_learnable=False):
        super(CEN_RCN, self).__init__()

        self.context_encoder = context_encoder

        self.fusion = nn.Linear(encoding_dim + n_features, n_hidden)
        self.input_encoder = nn.Linear(n_features, n_hidden)
        # self.fusion = nn.Sequential(
        #     nn.Linear(encoding_dim + n_features, n_hidden),
        #     nn.ReLU(True),
        #     nn.Linear(n_hidden, n_hidden)
        # )
        # self.input_encoder = nn.Sequential(
        #     nn.Linear(n_features, n_hidden),
        #     nn.ReLU(True),
        #     nn.Linear(n_hidden, n_hidden)
        # )

        self.mask_u = mask_u

        # TODO: change pyz to log pyz
        print(f"pyz_learnable {pyz_learnable}")
        # self.log_pyz = nn.Parameter(pyz.log(), requires_grad=pyz_learnable)
        self.pyz = nn.Parameter(pyz, requires_grad=pyz_learnable)
        
    def forward(self, context, x, S):
        # dot-product attention between rule representations and
        # fused rep of context and attributes

        # input x: n_train x n_features
        # x_rep: n_train x n_hidden
        # S: n_rules x n_train
        # S_rep: n_rules x n_hidden
        # h: n_train x n_hidden

        phi = self.context_encoder(context)
        h = self.fusion(torch.cat([phi, x], dim=-1))

        x_rep = self.input_encoder(x)
        # x_rep = phi
        S_rep = torch.matmul(S.transpose(0, 1), x_rep)

        # u: n_train x n_rules
        # scores for each
        u = torch.matmul(h, S_rep.transpose(0, 1))

        if self.mask_u:
            # only allow rules that the datapoint satisfies. We use S to mask
            # assumes that each datapoint satisfies at least one antecedent

            # print(S.sum(-1))
            # if not S.sum(-1).min() > 0:
            #     print("no antecedents satisfy datapoint!")
            #     raise ValueError

            if not S.sum(-1).min() > 0:
                pz = F.softmax(torch.ones_like(u), dim=-1)
            else:
                ninf = torch.full_like(u, np.NINF)
                masked_u = torch.where(S > 0, u, ninf)
                pz = F.softmax(masked_u, dim=-1)
            # pz = F.log_softmax(masked_u, dim=-1)
        
        else:
            pz = F.softmax(u, dim=-1)
            # pz = F.log_softmax(u, dim=-1)

        # sum_py = self.log_pyz[None, :, :] + pz[:, :, None]
        # py = sum_py.logsumexp(1)

        # TODO: add sequence transition parameters
        py = torch.matmul(pz, self.pyz)

        return pz, py

class FF(nn.Module):
    def __init__(self, context_encoder, encoding_dim, output_dim):
        """
        Directly predict p(y|x)
        """
        super(FF, self).__init__()

        self.context_encoder = context_encoder

        self.final = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(encoding_dim, output_dim)
        )

    def forward(self, x):
        phi = self.context_encoder(x)
        return F.softmax(self.final(phi), dim=-1)

def rae(y_true, y_pred):
    n_classes = y_true.shape[-1]

    expected_death = (torch.arange(n_classes, dtype=torch.float) * py).sum(-1)
    diff = expected_death - batch_classes.type_as(expected_death)
    rae = torch.min(torch.tensor(1.), torch.abs(diff / expected_death))

def create_model(args, pyz):
    encoding_dim = args['encoding_dim']

    n_features = args['n_features']

    encoder_args = {
        'n_features': n_features,
        'n_hidden': args['n_encoder_hidden'],
        'encoding_dim': encoding_dim
    }
    context_encoder = make_support2_encoder(encoder_args)

    # for now, x = c
    n_hidden = args['n_rcn_hidden']

    if args['model'] == 'rcn':
        model = CEN_RCN(context_encoder, encoding_dim, n_features, n_hidden,
                        pyz,
                        mask_u=args['rcn_mask'],
                        pyz_learnable=args['pyz_learnable'])
    elif args['model'] == 'ff':
        model = FF(context_encoder, encoding_dim, args['n_classes'])

    return model

def accuracy_score_self(y_true, y_pred, t):
    """
    y_true: batch_size x n_classes
    y_pred: 

    based on CEN implementation
    """

    censored_indicator = 1.

    y_true_c = y_true[:, :, 0]
    y_true_e = y_true[:, :, 1]

    # ignore censored values
    not_censored_at_t = np.not_equal(y_true_c[:, t], censored_indicator)
    y_true_uc = y_true_e[not_censored_at_t]
    y_pred_uc = y_pred[not_censored_at_t]

    # compute survival probabilities
    # y_pred[:, t] is the probability that the patient dies at time t
    # so cumulative sum gives us the probability that the patient dies by time t
    death_prob = np.minimum(np.cumsum(y_pred_uc, axis=-1), 1.)
    survival_prob = 1 - death_prob

    survived = 1 - y_true_uc[:, t]
    survival_pred = (survival_prob[:, t] > 0.5).astype(survived.dtype)

    acc = accuracy_score(survived, survival_pred)

    print(f"Acc@{t}: {acc:.4f}")

    return acc

def rae_loss_self(y_true, y_pred):
    """
    only calculate RAE for non-censored patients, like in CEN paper
    """
    censored_indicator = 1.

    # n = y_true.shape[0]
    # m = y_true.shape[1]

    not_censored = np.not_equal(y_true[:, -1, 0], censored_indicator)

    death_prob = np.minimum(np.cumsum(y_pred, axis=-1), 1.)
    pred_time = 1 + (death_prob > 0.5).astype('float32').argmax(-1)

    pred_time_nc = pred_time[not_censored]
    
    nc = pred_time_nc.shape[0]

    y_true_nc = y_true[:, :, 1][not_censored]

    # # add event for case where patient is still alive?
    # # otherwise the time will be 0/1, since all values are 0
    # y_true_nc = np.concatenate([y_true_nc, np.ones((y_true_nc.shape[0], 1))], axis=-1)

    # argmax returns first max index
    true_time_nc = 1 + y_true_nc.argmax(-1).astype('float32')

    # print(y_true_nc.sum(-1).min())
    # print(y_true_nc[y_true_nc.sum(-1).argmin()])
    # print(true_time_nc[y_true_nc.sum(-1).argmin()])
    # print('--')
    # print(y_true_nc[:5])
    # print(true_time_nc[:5])
    # sys.exit()

    loss_nc = np.abs(pred_time_nc - true_time_nc) / pred_time_nc
    rae_loss = loss_nc.mean()

    print(f"Mean RAE on non-censored: {rae_loss:.4f}\t({nc} not censored)")

    return rae_loss

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
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=1)

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

        if ep % 10 == 0 and valid_dataloader is not None:
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
                    
                    elif args['model'] == 'ff':
                        py = model(batch_c)

                    predictions = py.argmax(-1)
                    all_preds.append(predictions)
                    all_py.append(py)
                    valid_y.append(batch_sample['y'].numpy())

            all_preds = torch.cat(all_preds)
            all_py = torch.cat(all_py)
            valid_y = np.concatenate(valid_y).argmax(-1)
            valid_acc = accuracy_score(valid_y, all_preds.numpy())

            # RAE
            n_classes = all_py.size()[-1]
            expected_death = (np.arange(n_classes) * all_py.cpu().numpy()).sum(-1)
            diff = expected_death - valid_y
            valid_rae = np.minimum(1, np.abs(diff)).mean()

            print(all_preds[:10])
            print(valid_y[:10])
            print(diff[:10])

            log_line = f"Ep {ep:<5} -  log prob: {total_log_prob:.4f}, validation acc: {valid_acc:.4f}"
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
        accs[t] = accuracy_score_self(y_true_full, y_probs.cpu().numpy(), t)

    n_classes = y_probs.shape[-1]
    expected_death = (np.arange(n_classes) * y_probs.cpu().numpy()).sum(-1)
    diff = expected_death - test_classes
    test_rae = np.minimum(1, np.abs(diff)).mean()
    print(f"Test RAE: {test_rae:.4f}")

    rae_nc = rae_loss_self(y_true_full, y_probs.cpu().numpy())

    return accs, test_rae, rae_nc

def write_results(results, outfile):
    f = open(outfile, 'w')
    writer = csv.DictWriter(f, fieldnames=results.keys())

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['rcn', 'ff'])
    parser.add_argument('raw_file')
    parser.add_argument('categorical_file')
    parser.add_argument('label_file')

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

    return vars(parser.parse_args())

def main():
    """
    TODO:
    - subsample antecedents
    - add support for MNIST and IMDB
    """
    args = parse_arguments()

    train_data, valid_data, test_data, antes = load_data(args, 'support2')
    for d in [train_data, valid_data, test_data]:
        for k, v in d.items():
            if type(v) is list:
                print(k, len(v))
            else:
                print(k, v.shape)

        print('---')

    print(train_data['c'].shape)

    args['n_features'] = train_data['c'].shape[-1]

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

    train_dataset = Support2(train_data['x'], train_data['c'], train_data['y'], train_data['S'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    valid_dataset = Support2(valid_data['x'], valid_data['c'], valid_data['y'], valid_data['S'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_dataset = Support2(test_data['x'], test_data['c'], test_data['y'], test_data['S'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = train_model(args,
                        model,
                        train_loader,
                        valid_dataloader=valid_loader)

    all_preds, all_py, all_pz = pred_model(args, model, test_loader)

    accs, test_rae, rae_nc = compute_support2_metrics(args,
                                                    all_preds,
                                                    all_py,
                                                    test_data['y'],
                                                    test_data['y2'])

    test_classes = test_data['y'].argmax(-1)

    if args['model'] == 'rcn' and args['print_rules']:
        # all_pz = torch.cat(all_pz, dim=0)
        selected_antes = [antes[i] for i in all_pz.argmax(-1)]

        for i, (a, idx) in enumerate(zip(selected_antes, all_pz.argmax(-1))):
            print(a, pyz[idx])
            print(pyz[idx, test_classes[i]])
            if i >= 4:
                break

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

    if args['outfile']:
        with open(args['outfile'], 'a') as f:
            f.write(line)
            f.write('\n')

if __name__ == '__main__':
    main()