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


def predict(model, test, test_loader, is_flatten=False, is_TS=False):
    """
    Predict the missing values in the test set
    :param model: the model
    :param test: the test set
    :param test_loader: the test loader
    :param is_flatten: whether the tensor is flatten in the input of the model (MLP autoencoder case)
    :param is_TS: whether the model is a transformer
    :return: the reconstructed data and the one used as target
    """
    test_predicted = []
    test_or = []
    i = 1

    # reconstruct the data and impute the missing values
    with torch.no_grad():
        for target, masked_input, mask in test_loader:
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
            if is_TS:
                output = model(masked_input, None)
            else:
                output = model(masked_input)
            test_predicted.append(padding_tensor(output, test, i, is_flatten))
            test_or.append(padding_tensor(target, test, i, is_flatten))
            i += 1

    # calculate the element-wise sum of the prediction in the list then weigth average
    test_predicted_final = torch.zeros_like(test_predicted[0])
    for tensor in test_predicted:
        test_predicted_final = torch.add(test_predicted_final, tensor)

    weight_tensor = torch.zeros_like(test_predicted_final)
    for i in range(test_predicted_final.shape[1]):
        if i + 1 < target.shape[1]:
            weight_tensor[0][i] = test_predicted_final[0][i] / (i + 1)
        elif i + 1 > weight_tensor.shape[1] - target.shape[1] + 2:
            weight_tensor[0][i] = test_predicted_final[0][i] / (
                test_predicted_final.shape[1] - i
            )
        else:
            weight_tensor[0][i] = test_predicted_final[0][i] / target.shape[1]
    test_predicted_final = weight_tensor

    # calculate the element-wise sum of the used target in the list then weigth average
    test_or_final = torch.zeros_like(test_or[0])
    for tensor in test_or:
        test_or_final = torch.add(test_or_final, tensor)

    test_tensor = torch.zeros_like(test_or_final)
    for i in range(test_or_final.shape[1]):
        if i + 1 < target.shape[1]:
            test_tensor[0][i] = test_or_final[0][i] / (i + 1)
        elif i + 1 > test_or_final.shape[1] - target.shape[1] + 2:
            test_tensor[0][i] = test_or_final[0][i] / (test_or_final.shape[1] - i)
        else:
            test_tensor[0][i] = test_or_final[0][i] / target.shape[1]
    test_or_final = test_tensor

    # Reshape results
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

    # final dataframes
    test_predicted_final = pd.DataFrame(
        test_predicted_final[: len(test)].cpu().detach().numpy(), columns=test.columns
    )
    test_or_final = pd.DataFrame(
        test_or_final[: len(test)].cpu().detach().numpy(), columns=test.columns
    )
    return test_predicted_final, test_or_final
