import torch
import pandas as pd
from collections import defaultdict


def predict(
    model,
    test,
    sequence_length,
    test_loader,
    is_flatten=False,
    is_TS=False,
    strategy="median",
):
    """
    Predict the missing values in the test set
    :param model: the model
    :param test: the test set
    :param test_loader: the test loader
    :param is_flatten: whether the tensor is flatten in the input of the model (MLP autoencoder case)
    :param is_TS: whether the model is a transformer
    :return: the reconstructed data and the one used as target
    """
    # Init
    print("Prediction loop ...")
    test_predicted = defaultdict(lambda: [])
    test_or = defaultdict(lambda: [])
    i = 0
    len_test, number_features = test.shape
    # Prediction loop over loader
    for target, masked_input, mask in test_loader:
        if torch.cuda.is_available():
            target, masked_input, mask = target.cuda(), masked_input.cuda(), mask.cuda()
        if is_flatten:
            target = target.view(target.shape[0], target.shape[2] * target.shape[1])
            masked_input = masked_input.view(
                masked_input.shape[0], masked_input.shape[2] * masked_input.shape[1]
            )
        if is_TS:
            output = model(masked_input, None)
        elif is_flatten:
            output = model(masked_input)
            output = output.view(output.shape[0], sequence_length, number_features)
            target = target.view(target.shape[0], sequence_length, number_features)
        else:
            output = model(masked_input)
        k = 0
        for j in range(i, i + sequence_length):
            test_predicted[j].append(output[0][k])
            test_or[j].append(target[0][k])
            k += 1
        i += 1
    print("Prediction loop done")
    test_predicted_final = torch.zeros(len_test, number_features)
    test_or_final = torch.zeros(len_test, number_features)
    print("Aggregation ...")
    for i in range(len_test):
        if strategy == "mean":
            candidat = torch.sum(torch.stack(test_predicted[i]), dim=0)
            candidat_target = torch.sum(torch.stack(test_or[i]), dim=0)
            if i < sequence_length:
                change = candidat / (i + 1)
                change_target = candidat_target / (i + 1)
            elif i > (len_test % sequence_length) * sequence_length:
                change = candidat / (len_test - i)
                change_target = candidat_target / (len_test - i)
            else:
                change = candidat / sequence_length
                change_target = candidat_target / sequence_length
            test_predicted_final[i] = change
            test_or_final[i] = change_target
        elif strategy == "median":
            test_predicted_final[i] = torch.median(
                torch.stack(test_predicted[i]), dim=0
            ).values
            test_or_final[i] = torch.median(torch.stack(test_or[i]), dim=0).values
        else:
            raise ValueError(
                "Error : The strategy '{}' is not supported".format(strategy)
            )
    print("Aggregation done")
    # Final dataframes
    test_predicted_final = pd.DataFrame(
        test_predicted_final.cpu().detach().numpy(), columns=test.columns
    )
    test_or_final = pd.DataFrame(
        test_or_final.cpu().detach().numpy(), columns=test.columns
    )
    return test_predicted_final, test_or_final
