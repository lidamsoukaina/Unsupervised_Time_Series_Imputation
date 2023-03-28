import numpy as np
import pandas as pd
from scipy import stats


def t_test(
    df: pd.DataFrame, df_imputed: pd.DataFrame, mask: np.ndarray, method_title: str
) -> pd.DataFrame:
    """
    T-test for the imputed data and the original data
    :param df: original data
    :param df_imputed: imputed data
    :param mask: mask for the original data
    :param method_title: title of the imputation method
    :return: a dataframe with the p-value of the t-test for each column
    """
    df_imputed = df_imputed * mask + df * (1 - mask)
    ttest = stats.ttest_ind(df.values, df_imputed.values)
    test_stat = pd.DataFrame(
        columns=["method", "column", "p-value", "same_distribution"]
    )
    for n in range(len(ttest.pvalue)):
        test_stat.loc[n] = [
            method_title,
            df.columns[n],
            ttest.pvalue[n],
            ttest.pvalue[n] < 0.05,
        ]
    return test_stat
