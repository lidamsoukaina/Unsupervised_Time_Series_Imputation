import pandas as pd
from typing import Tuple, List


def preprocess_data(
    train_path: str,
    val_path: str,
    test_path: str,
    scaler: callable,
    columns_to_drop: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess Data
    :param train_path: Path to train data
    :param val_path: Path to validation data
    :param test_path: Path to test data
    :param scaler: Scaler to use
    :param columns_to_drop: list of columns to drop
    :return: Preprocessed train, validation and test data
    """
    # Read Data
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    # Drop unuseful columns
    train.drop(columns_to_drop, axis=1, inplace=True)
    val.drop(columns_to_drop, axis=1, inplace=True)
    test.drop(columns_to_drop, axis=1, inplace=True)
    # Scale Data
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    val = pd.DataFrame(scaler.transform(val), columns=val.columns)
    test = pd.DataFrame(scaler.transform(test), columns=test.columns)
    return train, val, test
