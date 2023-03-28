import pandas as pd
from typing import Literal
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from global_config import random_state


def classical_imputer(
    test: pd.DataFrame,
    strategy: Literal[
        "mean",
        "median",
        "mode",
        "LOCF",
        "NOCB",
        "linear_interpolation",
        "spline_interpolation",
        "knn",
        "mice",
    ],
    spline_order=3,
    random_state=random_state,
    n_neighbors=5,
) -> pd.DataFrame:
    """
    Impute missing values using classical imputation methods
    :param test: Dataframe to impute
    :param strategy: Imputation strategy
    :param spline_order: Order of spline interpolation
    :param random_state: Random state for mice imputation
    :param n_neighbors: Number of neighbors to be considered for knn imputation
    :return: Imputed dataframe
    """
    try:
        df = test.copy()
        df_imputed = None
        if strategy == "mean":
            mean_imp = SimpleImputer(strategy="mean")
            df_imputed = pd.DataFrame(mean_imp.fit_transform(df), columns=df.columns)
        elif strategy == "median":
            median_imp = SimpleImputer(strategy="median")
            df_imputed = pd.DataFrame(median_imp.fit_transform(df), columns=df.columns)
        elif strategy == "mode":
            most_frequent_imp = SimpleImputer(strategy="most_frequent")
            df_imputed = pd.DataFrame(
                most_frequent_imp.fit_transform(df), columns=df.columns
            )
        elif strategy == "LOCF":
            df_imputed = df.fillna(method="ffill").fillna(method="bfill")
        elif strategy == "NOCB":
            df_imputed = df.fillna(method="bfill").fillna(method="ffill")
        elif strategy == "linear_interpolation":
            df_imputed = df.interpolate(method="linear", limit_direction="both")
        elif strategy == "spline_interpolation":
            df_imputed = df.interpolate(
                method="spline", order=spline_order, limit_direction="both"
            )
        elif strategy == "knn":
            knn_imp = KNNImputer(n_neighbors=n_neighbors)
            df_imputed = pd.DataFrame(knn_imp.fit_transform(df), columns=df.columns)
        elif strategy == "mice":
            mice_imp = IterativeImputer(random_state=random_state)
            df_imputed = pd.DataFrame(mice_imp.fit_transform(df), columns=df.columns)
        else:
            raise ValueError(
                "Error : The strategy '{}' is not supported".format(strategy)
            )
        return df_imputed
    except ValueError as e:
        raise (e)
