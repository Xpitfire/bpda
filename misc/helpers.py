import random
import torch
import numpy as np
import os
import math
import importlib


def shift_data(x, y, shift=[0, 0], theta=0):
    """Data shift helper. Performs rotation and translation."""
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = np.tensordot(r, x, axes=([0], [1])) + np.array(shift)[:, np.newaxis]
    return x.T, y


def count_parameters(model):
    """Counts the available parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def import_from_file(model_path):
    """Import from file path"""
    import_file = os.path.basename(model_path).split(".")[0]
    import_root = os.path.dirname(model_path)
    imported = importlib.import_module("%s.%s" % (import_root, import_file))
    return imported


def import_from_package_by_name(object_name, package_root):
    """Import from package by name"""
    package = importlib.import_module(package_root)
    obj = getattr(package, object_name)
    return obj


def load_function(module, function):
    """Loads a function or class based on the module path and function argument"""
    module = os.path.splitext(module)[0].replace("/", ".")
    return import_from_package_by_name(function, module)


def map_reduce(items, key, redux=np.mean):
    """Remaps a dict to a list and reduces values"""
    items = [v[key] for v in items]
    return redux(items)


def load_seed_list(file_name='seed_list.txt'):
    """Loads seeds from a file"""
    with open(file_name, 'r') as f:
        seeds = f.read().split('\n')
    seeds = [int(s) for s in seeds if s is not None and s != '']
    return seeds


def set_seed(seed=None):
    """Sets the global seeds for random, numpy and torch"""
    if seed: random.seed(seed)
    if seed: np.random.seed(seed)
    if seed: torch.manual_seed(seed)
    if seed & torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    if seed & torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    r"""Early stops the training if validation loss doesn't improve after a given patience.
    Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_epochs (int): Forces training for a minimum ammount of time.
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
    """
    def __init__(self, trainer, patience=7, min_epochs=0, verbose=False, delta=0, trace_func=print):
        self.trainer = trainer
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, epoch, val_loss):
        if math.isnan(val_loss):
            print("WARNING! Received NaN values for the evaluation loss. Stopping Training.")
            self.early_stop = True
            return

        # ignore early stopping if minimum amount of epochs is not reached
        if epoch <= self.min_epochs:
            return

        score = -val_loss
        # check if already initialized, if not initilize
        if self.best_score is None:
            self.best_score = score
            self.trainer.save_checkpoint(epoch, 'best')
        # check if score has improved compared to previous version, if no update patient counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # set break criteria
            if self.counter >= self.patience:
                self.early_stop = True
        # otherwise save new best model
        else:
            self.best_score = score
            self.trainer.save_checkpoint(epoch, 'best')
            self.counter = 0
