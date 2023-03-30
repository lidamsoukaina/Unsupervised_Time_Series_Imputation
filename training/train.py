from .utils import NoamOpt
from IPython.display import clear_output
import datetime

import torch
from torch.optim.lr_scheduler import ExponentialLR


def train_model(
    model,
    loss_fn,
    config,
    input_dim,
    train_loader,
    val_loader,
    is_flatten,
    is_TS,
    is_warmed,
):
    """
    Train the model
    :param model: model to train
    :param loss_fn: loss function
    :param config: configuration file
    :param train_loader: train loader
    :param val_loader: validation loader
    :param is_flatten: if the sequence is flatten as for models with Linear layers at first layer
    :param is_TS: if the model is a transformer
    :param is_warmed: if the model is warmed up
    :return: train loss, validation loss, path of the saved model
    """
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    models_path = config["models_path"]
    # optimizer
    if is_warmed:
        optimizer = NoamOpt(
            input_dim,
            2,
            100,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        # training phase
        train_loss = 0
        model.train()
        for i, (target, masked_input, mask) in enumerate(train_loader):
            if torch.cuda.is_available():
                target, masked_input, mask = (
                    target.cuda(),
                    masked_input.cuda(),
                    mask.cuda(),
                )
            if is_flatten:
                target = target.view(target.shape[0], target.shape[2] * target.shape[1])
                masked_input = masked_input.view(
                    masked_input.shape[0], masked_input.shape[2] * masked_input.shape[1]
                )
                mask = mask.view(mask.shape[0], mask.shape[2] * mask.shape[1])
            if is_warmed:
                optimizer.optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            if is_TS:
                outputs = model(masked_input, None)
            else:
                outputs = model(masked_input)
            loss = loss_fn(outputs, target, mask)
            loss.backward()
            optimizer.step()
            # lr_values.append(model_opt.optimizer.param_groups[0]['lr'])
            train_loss += loss.item()
        if not is_warmed:
            scheduler.step()

        # validation phase
        valid_loss = 0.0
        model.eval()
        for i, (target, masked_input, mask) in enumerate(val_loader):
            if torch.cuda.is_available():
                target, masked_input, mask = (
                    target.cuda(),
                    masked_input.cuda(),
                    mask.cuda(),
                )
            if is_flatten:
                target = target.view(target.shape[0], target.shape[2] * target.shape[1])
                masked_input = masked_input.view(
                    masked_input.shape[0], masked_input.shape[2] * masked_input.shape[1]
                )
                mask = mask.view(mask.shape[0], mask.shape[2] * mask.shape[1])
            if is_TS:
                outputs = model(masked_input, None)
            else:
                outputs = model(masked_input)
            loss = loss_fn(outputs, target, mask)
            valid_loss += loss.item()
        train_loss_list.append(train_loss / len(train_loader))
        val_loss_list.append(valid_loss / len(val_loader))
        print(
            "Epoch {}: train loss: {}, val loss: {}".format(
                epoch, train_loss / len(train_loader), valid_loss / len(val_loader)
            )
        )
    # Saving State Dict
    date_time = datetime.datetime.now()
    index = ("_").join(str(date_time).split(" "))
    PATH = models_path + "/model_final_" + index
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "config_model": config,
        },
        PATH,
    )
    clear_output()
    return train_loss_list, val_loss_list, PATH
