from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

data = pd.read_csv("./data/household_power_consumption.csv")


def agglomerate_data(data, strategy="mean", frequence="H"):
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
