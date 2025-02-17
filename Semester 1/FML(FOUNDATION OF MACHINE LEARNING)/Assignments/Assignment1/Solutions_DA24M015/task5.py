import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_train.csv",header=None)
test_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_test.csv",header=None)

train_data.columns = ['Feature1', 'Feature2', 'Output']

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def rbf_kernel(X1, X2, gamma):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    return np.exp(-gamma * sq_dists)

def kernel_regression(X_train, y_train, X_test, gamma):
    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)

    K_train_inv = np.linalg.inv(K_train + np.eye(K_train.shape[0]) * 1e-5)
    alpha = K_train_inv @ y_train

    y_pred = K_test @ alpha
    return y_pred


gamma = 0.5
y_pred = kernel_regression(X_train, y_train, X_test, gamma)
test_error_kernel = mean_squared_error(y_test, y_pred)
w_ml = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
y_pred_OLS= X_test @ w_ml
test_error_OLS = mean_squared_error(y_test, y_pred_OLS)

print(f"Test Error for Kernel Ridge Regression (RBF): {test_error_kernel}")
print(f"Test Error for Ordinary Least Squares (wML): {test_error_OLS}")