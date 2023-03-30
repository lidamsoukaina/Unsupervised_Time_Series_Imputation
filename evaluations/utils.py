import torch


def load_model(model, model_path):
    """
    Load model from checkpoint
    :param model: model to load
    :param model_path: path to checkpoint
    :return: model
    """
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    return model


def count_parameters(model):
    """
    Count the number of parameters in a model
    :param model: model to count parameters
    :return: number of parameters
    """
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return nb_params
