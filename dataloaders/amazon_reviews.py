from torch.utils.data import Dataset

import torch
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

DOMAIN_DICT = {'books': 0, 'dvd': 1, 'electronics': 2, 'kitchen': 3}

class AmazonReviewsDataset(Dataset):
    r"""Domain adaptation version of the moon dataset object to iterate and collect samples.
    """
    def __init__(self, data):
        self.xs, self.ys, self.xt, self.yt = data

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        xs = self.xs[idx]
        ys = self.ys[idx]
        xt = self.xt[idx]
        yt = self.yt[idx]
        # convert to tensors
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(np.array(ys).astype(np.int64))
        xt = torch.from_numpy(xt.astype(np.float32))
        yt = torch.from_numpy(np.array(yt).astype(np.int64))
        return xs, ys, xt, yt

def split_data(d_s_ind, d_t_ind, x, y, offset, n_tr_samples, r_seed=0):
    """
    split data (train/validation/test, source/target)
    """

    # setting the random seed, for reproducibility
    np.random.seed(r_seed)

    # creating source and target data folds.
    x_s_tr = x[offset[d_s_ind, 0]:offset[d_s_ind, 0] + n_tr_samples, :]
    x_t_tr = x[offset[d_t_ind, 0]:offset[d_t_ind, 0] + n_tr_samples, :]
    x_s_tst = x[offset[d_s_ind, 0] + n_tr_samples:offset[d_s_ind + 1, 0], :]
    x_t_tst = x[offset[d_t_ind, 0] + n_tr_samples:offset[d_t_ind + 1, 0], :]
    y_s_tr = y[offset[d_s_ind, 0]:offset[d_s_ind, 0] + n_tr_samples]
    y_t_tr = y[offset[d_t_ind, 0]:offset[d_t_ind, 0] + n_tr_samples]
    y_s_tst = y[offset[d_s_ind, 0] + n_tr_samples:offset[d_s_ind + 1, 0]]
    y_t_tst = y[offset[d_t_ind, 0] + n_tr_samples:offset[d_t_ind + 1, 0]]

    # setting labels to 0 and 1
    y_s_tr[y_s_tr == -1] = 0
    y_t_tr[y_t_tr == -1] = 0
    y_s_tst[y_s_tst == -1] = 0
    y_t_tst[y_t_tst == -1] = 0

    y_s_tr = np.squeeze(y_s_tr, axis=1)
    y_t_tr = np.squeeze(y_t_tr, axis=1)
    y_s_tst = np.squeeze(y_s_tst, axis=1)
    y_t_tst = np.squeeze(y_t_tst, axis=1)

    return x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst


def create_domain_adaptation_data(config):
    data = scipy.io.loadmat(config.dataloader.AmazonReviews.filename)
    x = data['xx'][:config.dataloader.AmazonReviews.n_features, :].toarray().T
    y = data['yy']
    offset = data['offset']
    x_s_tr, y_s_tr, x_t_tr, y_t_tr, Xs_test, Ys_test, Xt_test, Yt_test = split_data(DOMAIN_DICT[config.dataloader.AmazonReviews.source_domain_ix],
                                                                                    DOMAIN_DICT[config.dataloader.AmazonReviews.target_domain_ix],
                                                                                    x, y,
                                                                                    offset, config.dataloader.AmazonReviews.n_tr_samples,
                                                                                    config.dataloader.AmazonReviews.r_seed)

    Xs_train, Xs_eval, Ys_train, Ys_eval = train_test_split(x_s_tr, y_s_tr, test_size=0.3, random_state=config.dataloader.AmazonReviews.r_seed)
    Xt_train, Xt_eval, Yt_train, Yt_eval = train_test_split(x_t_tr, y_t_tr, test_size=0.3, random_state=config.dataloader.AmazonReviews.r_seed)

    if len(Ys_test) > len(Yt_test):
        Xs_test = Xs_test[:Yt_test.shape[0]]
        Ys_test = Ys_test[:Yt_test.shape[0]]
    else:
        Yt_test = Yt_test[:Ys_test.shape[0]]
        Yt_test = Yt_test[:Ys_test.shape[0]]

    if config.dataloader.AmazonReviews.loading_schema == 'train-eval':
        train_loader = torch.utils.data.DataLoader(
            AmazonReviewsDataset((Xs_train, Ys_train, Xt_train, Yt_train)),
            batch_size=config.dataloader.AmazonReviews.batchsize,
            shuffle=True
        )
        eval_loader = torch.utils.data.DataLoader(
            AmazonReviewsDataset((Xs_eval, Ys_eval, Xt_eval, Yt_eval)),
            batch_size=config.dataloader.AmazonReviews.batchsize,
            shuffle=False
        )
        return (train_loader, eval_loader), (Xs_train, Xs_eval, Ys_train, Ys_eval, Xt_train, Xt_eval, Yt_train, Yt_eval)
    elif config.dataloader.AmazonReviews.loading_schema == 'train-test':
        train_loader = torch.utils.data.DataLoader(
            AmazonReviewsDataset((Xs_train, Ys_train, Xt_train, Yt_train)),
            batch_size=config.dataloader.AmazonReviews.batchsize,
            shuffle=True
        )
        eval_loader = torch.utils.data.DataLoader(
            AmazonReviewsDataset((Xs_eval, Ys_eval, Xt_eval, Yt_eval)),
            batch_size=config.dataloader.AmazonReviews.batchsize,
            shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            AmazonReviewsDataset((Xs_test, Ys_test, Xt_test, Yt_test)),
            batch_size=config.dataloader.AmazonReviews.batchsize,
            shuffle=False
        )
        return (train_loader, eval_loader, test_loader), (Xs_train, Xs_eval, Xs_test, Ys_train, Ys_eval, Ys_test, \
                                                        Xt_train, Xt_eval, Xt_test, Yt_train, Yt_eval, Yt_test)
    elif config.dataloader.AmazonReviews.loading_schema == 'test':
        test_loader = torch.utils.data.DataLoader(
            AmazonReviewsDataset((Xs_test, Ys_test, Xt_test, Yt_test)),
            batch_size=config.dataloader.AmazonReviews.batchsize,
            shuffle=False
        )
        return (test_loader), (Xs_test, Ys_test, Xt_test, Yt_test)
