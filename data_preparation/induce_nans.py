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


def generate_MAR_nans(df, col1, col2, missing_ratio=0.1, random_state=42):
    """
    Generate Missing values in MAR way
    :param df: Dataframe to be masked
    :param col1: Column to be used as a condition
    :param col2: Column to be masked
    :param missing_ratio: Missing ratio per each feature
    :param random_state: Random state
    :return: Dataframe with missing values and mask
    """
    df_nan = df.copy()
    np.random.seed(random_state)
    missing_mask = (
        np.random.uniform(
            0,
            1,
            df.shape[0],
        )
        < missing_ratio
    )
    missing_mask &= df[col1] < np.percentile(df[col1], 50)
    df_nan.loc[missing_mask, col2] = np.nan
    mask = np.isnan(df_nan.values).astype(int)
    return df_nan, mask


def generating_missing_values(df, missing_ratio, random_state, col1, col2):
    """
    Generate Missing values in MCAR and MAR way
    :param df: Dataframe to be masked
    :param missing_ratio: Missing ratio per each feature
    :param random_state: Random state
    :param col1: Column to be used as a condition
    :param col2: Column to be masked
    :return: Dataframe with missing values and mask
    """
    df_nan, mask = generate_MCAR_nans(df, missing_ratio, random_state=random_state)
    df_nan_mar, missing_mask = generate_MAR_nans(
        df_nan, col1, col2, missing_ratio, random_state=random_state
    )
    final_mask = np.add(mask, missing_mask)
    final_mask[final_mask >= 1] = 1
    return (df_nan, final_mask)


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
