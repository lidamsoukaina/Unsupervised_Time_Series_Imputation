import numpy as np
import pandas as pd
from typing import Tuple
from .mse import evaluate_imputation_mse
from .t_test import t_test
from models.baseline_models import classical_imputer


def evaluate_set(
    list_methods: list,
    test: pd.DataFrame,
    test_nan: pd.DataFrame,
    test_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the imputation methods on the test set
    :param list_methods: list of imputation methods
    :param test: test set
    :param test_nan: test set with missing values
    :param test_mask: mask of missing values
    :return: a tuple of two dataframes, the first one contains the mse of each imputation method, the second one contains the t-test results
    """
    evals = pd.DataFrame()
    tests = pd.DataFrame()
    for strat in list_methods:
        df_imputed = classical_imputer(test_nan, strat)
        evals = pd.concat(
            [evals, evaluate_imputation_mse(test, df_imputed, test_mask, strat)]
        )
        tests = pd.concat([tests, t_test(test, df_imputed, test_mask, strat)])
    evals.sort_values(by=["mse"], inplace=True)
    evals.reset_index(drop=True, inplace=True)
    return evals, tests
