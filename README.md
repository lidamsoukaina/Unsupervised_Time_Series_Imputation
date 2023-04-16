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
mkdir trained_models
mkdir data trained_models/AE trained_models/convAE trained_models/LSTM_AE trained_models/TS
```
6. Add the csv file 'household_power_consumption.csv' to `data` folder (link to the csv https://drive.google.com/drive/folders/10OYuhaT3nEaJmoGJLNMzOiSVPCtMJJtW?usp=sharing)

**Remark**:
If you want to use your own dataset, you need to add it to the `data` folder as 3 csv file (train, val and test) and edit the `config.yaml` file.
Eding the `config.yaml` file is necessary to specify the path to the csv files and the name of the unnecessary columns (if none: columns_to_drop: []) .

7. Now you can run the `main.ipynb` notebook.

**Remark**:
- If you are using you own dataset, run the `test.ipynb` notebook.
- You can change the hyperparameters of each model in the config files contained in the `training` folder.
## :tada: Implemented models
As baseline, we implemented various statistical based models : Linear Interpolation, MICE, NOCB, LOCF, Spline Interpolation, Median and Mode.

To assess the effectiveness of using deep learning models for the task of unsupervised time series imputation, we implemented four different architectures in a try to cover the main types of neural network:
- [x] Autoencoder
- [x] Convolutional Autoencoder
- [x] LSTM Autoencoder
- [x] Transformer Encoder

The architecture of the models is described in the `models` folder.

## :pencil2: Authors
- LETAIEF Maram
- LIDAM Soukaina
