import torch
import torch.nn as nn


# helper map to ease the switching of activation functions
activations = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'selu': nn.SELU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


def predict_from_logits(logits):
    """Takes logits and retrieves max predictions"""
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    return pred
