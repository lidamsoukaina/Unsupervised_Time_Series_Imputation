import torch
import torch.nn as nn
import copy


def clones(module, N):
    """Produce N identical layers
    :param module: the module to be cloned
    :param N: the number of clones
    :return: a list of N identical modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    :param model: the model
    :return: the number of trainable parameters
    """
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return nb_params
