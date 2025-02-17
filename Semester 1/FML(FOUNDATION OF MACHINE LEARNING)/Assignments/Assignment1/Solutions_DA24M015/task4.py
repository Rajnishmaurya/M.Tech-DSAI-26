import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_train.csv",header=None)
test_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_test.csv",header=None)

data.columns = ['Feature1', 'Feature2', 'Output']

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def ridge_gradient_descent(X_train, y_train, X_val, y_val, learning_rate, iterations, lambda_val):
    w = np.zeros(X_train.shape[1])
    n = X_train.shape[0]
    history = []

    for _ in range(iterations):
        y_pred = X_train @ w

        gradient = (2 / n) * X_train.T @ (y_pred - y_train) + lambda_val * w

        w = w - learning_rate * gradient

        y_val_pred = X_val @ w
        val_error = np.mean((y_val - y_val_pred) ** 2)
        history.append(val_error)

    return w, history

learning_rate = 0.01
iterations = 1000
lambda_values = [0.0001,0.001,0.01, 0.1, 1, 10, 100]

validation_errors = []
best_lambda = None
best_w = None

for lambda_val in lambda_values:
    _, history = ridge_gradient_descent(X_train, y_train, X_val, y_val, learning_rate, iterations, lambda_val)
    validation_errors.append(history[-1])

    if best_lambda is None or history[-1] < min(validation_errors):
        best_lambda = lambda_val
        best_w = _

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, validation_errors, marker='o')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Validation Error')
plt.title('Validation Error vs. Lambda for Ridge Regression')
plt.grid(True)
plt.show()

print("Best Lambda:", best_lambda)


X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def compute_test_error(X_test, y_test, w_R):
    y_test_pred = X_test @ w_R
    test_error = np.mean((y_test - y_test_pred) ** 2)
    return test_error

test_error_ridge = compute_test_error(X_test, y_test, best_w)
print("Test Error for Ridge Regression:", test_error_ridge)

w_ML = np.linalg.inv(X.T @ X) @ (X.T @ y)
test_error_ML = compute_test_error(X_test, y_test, w_ML)
print("Test Error for Least Squares:", test_error_ML)

if test_error_ridge < test_error_ML:
    print("Ridge regression with lambda =", best_lambda, "performs better on the test dataset.")
else:
    print("Least squares regression performs better on the test dataset.")
