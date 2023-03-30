from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configue


def agglomerate_data(
    path: str, strategy: str = "mean", frequence: str = "H"
) -> pd.DataFrame:
    """
    Agglomerate data by a given strategy and frequence
    :param path: path to the data
    :param strategy: strategy to agglomerate data
    :param frequence: frequence to agglomerate data
    :return: agglomerated data
    """
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(
        data["Date"] + data["Time"], format="%Y-%m-%d%H:%M:%S"
    )
    data.drop(columns=["Time"], inplace=True)
    if strategy == "mean":
        df = data.resample(frequence, on="Date").mean()
    elif strategy == "sum":
        df = data.resample(frequence, on="Date").sum()
    else:
        assert strategy == "sum" or strategy == "mean"
    return df


def create_train_val_test_sets(
    data: pd.DataFrame,
    train_size: float,
    test_size: float,
    random_state: int,
    path_train: str,
    path_val: str,
    path_test: str,
):
    """
    Split data into train, validation and test sets
    :param data: data to split
    :param train_size: size of the train set
    :param test_size: size of the test set
    :return: train, validation and test sets
    """
    df, test = train_test_split(
        data, shuffle=False, test_size=test_size, random_state=random_state
    )
    train, val = train_test_split(
        df, shuffle=False, train_size=train_size, random_state=random_state
    )
    train.dropna(inplace=True)
    val.dropna(inplace=True)
    test.dropna(inplace=True)
    test.to_csv(path_test)
    train.to_csv(path_train)
    val.to_csv(path_val)
    return train, val, test


def main(config: dict):
    data = agglomerate_data(config["path_data"])
    train, val, test = create_train_val_test_sets(
        data,
        config["train_size"],
        config["test_size"],
        config["random_state"],
        config["path_train"],
        config["path_val"],
        config["path_test"],
    )
    print("Train, validation and test sets created")


if __name__ == "__main__":
    config = configue.load("./config.yaml")
    main(config)
