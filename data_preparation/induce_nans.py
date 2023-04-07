import numpy as np
import pandas as pd
from typing import Tuple
import torch
import random


def generate_MCAR_nans(
    df: pd.DataFrame, missing_ratio: int, random_state: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate Missing values in MCAR way
    :param df: Dataframe to be masked
    :param missing_ratio: Missing ratio per each feature
    :param random_state: Random state
    :return: Dataframe with missing values and mask
    """
    df_nan = df.copy()
    mask_orig_test = df.isnull().astype(int).to_numpy()
    mask = np.zeros(df.shape)
    random.seed(random_state)
    mask_indices = [
        (i, j)
        for i in range(mask_orig_test.shape[0])
        for j in range(mask_orig_test.shape[1])
    ]
    random.shuffle(mask_indices)
    mask_indice_reduced = mask_indices[: int(len(df) * missing_ratio)]
    for (i, j) in mask_indice_reduced:
        df_nan.iloc[i, j] = np.nan
        mask[i, j] = 1
    return df_nan, mask


def random_mask_tensor(
    tensor: torch.Tensor, missing_ratio: float, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly masks a tensor with a given missing ratio.
    :param tensor: the tensor to be masked
    :param missing_ratio: The ratio of values to be masked
    :return: the masked tensor and the mask tensor
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Init masked tensor
    masked_input = tensor.detach()
    # Original Mask
    mask_orig = torch.isnan(tensor)
    # Generate artificially masked tensor avoiding previously masked elements
    intermidiate_tensor = torch.where(
        ~mask_orig, torch.rand(tensor.shape), torch.ones(tensor.shape)
    )
    mask_artif = intermidiate_tensor < missing_ratio
    # Mask tensor
    masked_input[mask_artif] = float("nan")
    # Return the masked tensor and the mask tensor
    return masked_input, mask_artif, mask_orig


def main(config: dict, test: pd.DataFrame):
    test_nan, test_mask = generate_MCAR_nans(
        test, config["missing_ratio"], config["random_state"]
    )
    print("Test data masked")
    return test_nan, test_mask
