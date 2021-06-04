import torch
import numpy as np
from torch.utils.data import Dataset

class DomainAdaptationMoonDataset(Dataset):
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


def create_domain_adaptation_data(config):
    """Creates a domain adaptation version of the moon datasets and dataloader"""
    # load data from file

    Xs_train = np.load(config.dataloader.MoonsNS.source_train_x)
    Ys_train = np.argmax(np.load(config.dataloader.MoonsNS.source_train_y), axis=1)
    Xt_train = np.load(config.dataloader.MoonsNS.target_train_x)
    Yt_train = np.argmax(np.load(config.dataloader.MoonsNS.target_train_y), axis=1)

    Xs_eval = np.load(config.dataloader.MoonsNS.source_valid_x)
    Ys_eval = np.argmax(np.load(config.dataloader.MoonsNS.source_valid_y), axis=1)
    Xt_eval = np.load(config.dataloader.MoonsNS.target_valid_x)
    Yt_eval = np.argmax(np.load(config.dataloader.MoonsNS.target_valid_y), axis=1)

    Xs_test = np.load(config.dataloader.MoonsNS.source_test_x)
    Ys_test = np.argmax(np.load(config.dataloader.MoonsNS.source_test_y), axis=1)
    Xt_test = np.load(config.dataloader.MoonsNS.target_test_x)
    Yt_test = np.argmax(np.load(config.dataloader.MoonsNS.target_test_y), axis=1)


    if config.dataloader.MoonsNS.loading_schema == 'train-eval':
        train_loader = torch.utils.data.DataLoader(
            DomainAdaptationMoonDataset((Xs_train, Ys_train, Xt_train, Yt_train)),
            batch_size=config.trainer.batchsize,
            shuffle=True
        )
        eval_loader = torch.utils.data.DataLoader(
            DomainAdaptationMoonDataset((Xs_eval, Ys_eval, Xt_eval, Yt_eval)),
            batch_size=config.trainer.batchsize,
            shuffle=False
        )
        return (train_loader, eval_loader), (Xs_train, Xs_eval, Ys_train, Ys_eval, Xt_train, Xt_eval, Yt_train, Yt_eval)
    elif config.dataloader.MoonsNS.loading_schema == 'train-test':
        train_loader = torch.utils.data.DataLoader(
            DomainAdaptationMoonDataset((Xs_train, Ys_train, Xt_train, Yt_train)),
            batch_size=config.trainer.batchsize,
            shuffle=True
        )
        eval_loader = torch.utils.data.DataLoader(
            DomainAdaptationMoonDataset((Xs_eval, Ys_eval, Xt_eval, Yt_eval)),
            batch_size=config.trainer.batchsize,
            shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            DomainAdaptationMoonDataset((Xs_test, Ys_test, Xt_test, Yt_test)),
            batch_size=config.trainer.batchsize,
            shuffle=False
        )
        return (train_loader, eval_loader, test_loader), (Xs_train, Xs_eval, Xs_test, Ys_train, Ys_eval, Ys_test, \
                                                        Xt_train, Xt_eval, Xt_test, Yt_train, Yt_eval, Yt_test)
    elif config.dataloader.MoonsNS.loading_schema == 'test':
        test_loader = torch.utils.data.DataLoader(
            DomainAdaptationMoonDataset((Xs_test, Ys_test, Xt_test, Yt_test)),
            batch_size=config.trainer.batchsize,
            shuffle=False
        )
        return (test_loader), (Xs_test, Ys_test, Xt_test, Yt_test)
