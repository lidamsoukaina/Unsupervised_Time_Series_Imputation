<div align="center">

# Unsuperised Time Series Imputation
[![Code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

</div>

## ❓ Context
A common challenge encountered during working with time series is the presence of missing data.
To address this issue, imputation is a widely used approach that involves filling in missing values rather than dropping them. However, the key challenge in imputation is determining the appropriate values to use for filling in the missing data.

## 🎯 Objective
In this project, we propose to assess the effectiveness of applying deep learning-based models in time series imputation compared to statistical methods that do not require prior training.

## :rocket: How to use the project

1. First, you need to clone the repository and `cd` into it :
```bash
git clone https://github.com/lidamsoukaina/Unsupervised_Time_Series_Imputation.git
cd Unsupervised_Time_Series_Imputation
```
2. Then, you need to create a virtual environment and activate it :
```bash
python3 -m venv venv
source venv/bin/activate
```
3. You need to install all the `requirements` using the following command :
```bash
pip install -r requirements.txt
```
4. [Optional] if you are using this repository in development mode, you can run the following command to set up the git hook scripts:
```bash
pre-commit install
```
5. You need to create folders `data` and `trained_models` and some subfolders:
```bash
mkdir data trained_models/AE trained_models/convAE trained_models/LSTM_AE trained_models/TS
```
6. Add your data in `data` folder (to test our used datasets, it is accessible through the link: https://drive.google.com/drive/folders/10OYuhaT3nEaJmoGJLNMzOiSVPCtMJJtW?usp=sharing)

remark: The data is a csv file of name 'household_power_consumption.csv' to be placed in the `data` folder

7. edit the config file.

To test the project, you can run the `main.ipynb` notebook.

## :memo: Results
TODO: List models results compared to baseline models


| model | MSE | MEAN | STD |
|---|---|---|---|
| LOCF | XX | XX | XX |
| Transformer | XX | XX | XX |

TODO: Describe and analyse the results

## 🤔 What's next ?
TODO: List the next steps

## :pencil2: Authors
- LETAIEF Maram
- LIDAM Soukaina
