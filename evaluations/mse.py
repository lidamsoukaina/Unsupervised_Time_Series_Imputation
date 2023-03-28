import numpy as np
import pandas as pd
from typing import Tuple


def evaluate_imputation_mse(
    df: pd.DataFrame, df_imputed: pd.DataFrame, mask: np.ndarray, method_title: str
) -> pd.DataFrame:
    """
    Evaluate the imputation performance by calculating the mean squared error (MSE) between the imputed and the original data.
    :param df: original data
    :param df_imputed: imputed data
    :param mask: mask of the missing values
    :param method_title: title of the imputation method
    :return: pd.DataFrame with the MSE
    """
    mse_per_column = np.sum((df * mask - df_imputed * mask) ** 2) / np.sum(mask, axis=0)
    result = pd.DataFrame(
        {
            "method": method_title,
            "mse": mse_per_column.mean(),
            "std": mse_per_column.std(),
        },
        index=[0],
    )
    return result
