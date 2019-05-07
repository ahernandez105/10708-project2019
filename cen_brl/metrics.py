import numpy as np

from sklearn.metrics import accuracy_score

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
