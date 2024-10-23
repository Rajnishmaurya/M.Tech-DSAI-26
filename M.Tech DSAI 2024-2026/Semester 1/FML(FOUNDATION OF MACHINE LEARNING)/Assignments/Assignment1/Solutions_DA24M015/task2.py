import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_train.csv",header=None)
test_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_test.csv",header=None)

train_data.columns = ['Feature1', 'Feature2', 'Output']

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

w_ML = np.linalg.inv(X.T @ X) @ (X.T @ y)

def gradient_descent(X, y, w_initial, learning_rate, iterations):
    w = w_initial
    n = X.shape[0]
    past_history = []

    for _ in range(iterations):
        y_pred = X @ w
        gradient = (2/n) * X.T @ (y_pred - y)
        w = w - learning_rate * gradient
        norm_diff = np.linalg.norm(w - w_ML)**2
        past_history.append(norm_diff)

    return w, past_history

learning_rate = 0.001
iterations = 10000
w_initial = np.zeros(X.shape[1])

w_final, past_history = gradient_descent(X, y, w_initial, learning_rate, iterations)

plt.figure(figsize=(10, 6))
plt.plot(range(iterations), past_history, label='∥wt − wML∥2')
plt.xlabel('Iteration')
plt.ylabel('Squared Norm of Difference')
plt.title('Convergence of Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
print("Least squares solution wML:", w_ML)
print("Least square solution w_final:",w_final)