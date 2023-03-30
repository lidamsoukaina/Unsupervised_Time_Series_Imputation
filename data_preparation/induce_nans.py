import numpy as np
import pandas as pd
from typing import Tuple
import torch


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
    np.random.seed(random_state)
    df_nan = df.mask(np.random.random(df.shape) < missing_ratio, inplace=False)
    mask = np.isnan(df_nan.values).astype(int)
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
    # Calculate the number of values to mask
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    num_masked_values = int(tensor.numel() * missing_ratio)
    # Create a mask tensor with the same shape as the input tensor
    mask = torch.ones_like(tensor)
    # Randomly select values to be masked and set their corresponding mask tensor values to 0
    mask_indices = torch.randperm(tensor.numel())[:num_masked_values]
    mask = mask.flatten()
    mask[mask_indices] = 0
    mask = mask.reshape(tensor.shape)
    mask = 1 - mask
    mask = mask.bool()
    # Mask the input tensor
    masked_tensor = tensor.masked_fill(mask, 0)
    # Return the masked tensor and the mask tensor
    return masked_tensor, mask


def main(config: dict, test: pd.DataFrame):
    test_nan, test_mask = generate_MCAR_nans(
        test, config["missing_ratio"], config["random_state"]
    )
    print("masking done")
    return test_nan, test_mask
