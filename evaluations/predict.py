import torch
import pandas as pd


def padding_tensor(tensor, len_test, number_features, index, is_flatten):
    """
    Pad the tensor with zeros
    :param tensor: the tensor to be padded
    :param df: the original dataframe
    :param index: the index of the tensor in the dataframe with regard to the sliding window
    :param is_flatten: whether the tensor is flatten
    :return: the padded tensor
    """
    if is_flatten:
        padding_left = torch.zeros(tensor.shape[0], index * number_features)
        padding_right = torch.zeros(
            tensor.shape[0],
            len_test * number_features - tensor.shape[1] - index * number_features,
        )
    else:
        padding_left = torch.zeros(tensor.shape[0], index, number_features)
        padding_right = torch.zeros(
            tensor.shape[0], len_test - tensor.shape[1] - index, number_features
        )
    if torch.cuda.is_available():
        padding_left, padding_right = padding_left.cuda(), padding_right.cuda()
    result = torch.cat([padding_left, tensor, padding_right], dim=1)
    return result


def predict(model, test, sequence_length, test_loader, is_flatten=False, is_TS=False):
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
    test_predicted = []
    test_or = []
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
        else:
            output = model(masked_input)
        ## Padding
        test_predicted.append(
            padding_tensor(output, len_test, number_features, i, is_flatten)
        )
        test_or.append(padding_tensor(target, len_test, number_features, i, is_flatten))
        i += 1

    # Weighted sum of predictions and target
    test_predicted_final = torch.zeros_like(test_predicted[0])
    for tensor in test_predicted:
        test_predicted_final = torch.add(test_predicted_final, tensor)

    test_or_final = torch.zeros_like(test_or[0])
    for tensor in test_or:
        test_or_final = torch.add(test_or_final, tensor)

    test_predicted_temp = torch.zeros_like(test_predicted_final)
    test_or_temp = torch.zeros_like(test_or_final)

    for i in range(len_test):
        if is_flatten:
            candidat = test_predicted_final[0][
                i * number_features : (i + 1) * number_features
            ]
            candidat_target = test_or_final[0][
                i * number_features : (i + 1) * number_features
            ]
        else:
            candidat = test_predicted_final[0][i]
            candidat_target = test_or_final[0][i]
        if i < sequence_length:
            change = candidat / (i + 1)
            change_target = candidat_target / (i + 1)
        elif i > (len_test % sequence_length) * sequence_length:
            change = candidat / (len_test - i)
            change_target = candidat_target / (len_test - i)
        else:
            change = candidat / sequence_length
            change_target = candidat_target / sequence_length
        if is_flatten:
            test_predicted_temp[0][
                i * number_features : (i + 1) * number_features
            ] = change
            test_or_temp[0][
                i * number_features : (i + 1) * number_features
            ] = change_target
        else:
            test_predicted_temp[0][i] = change
            test_or_temp[0][i] = change_target

    test_predicted_final = test_predicted_temp
    test_or_final = test_or_temp

    # Reshape
    if is_flatten:
        test_predicted_final = test_predicted_final.view(test.shape[0], test.shape[1])
        test_or_final = test_or_final.view(test.shape[0], test.shape[1])
    else:
        test_predicted_final = torch.reshape(
            test_predicted_final,
            (
                test_predicted_final.shape[0] * test_predicted_final.shape[1],
                test_predicted_final.shape[2],
            ),
        )
        test_or_final = torch.reshape(
            test_or_final,
            (test_or_final.shape[0] * test_or_final.shape[1], test_or_final.shape[2]),
        )

    # Final dataframes
    test_predicted_final = pd.DataFrame(
        test_predicted_final[: len(test)].cpu().detach().numpy(), columns=test.columns
    )
    test_or_final = pd.DataFrame(
        test_or_final[: len(test)].cpu().detach().numpy(), columns=test.columns
    )
    return test_predicted_final, test_or_final
