import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_train.csv",header=None)
test_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_test.csv",header=None)

train_data.columns = ['Feature1', 'Feature2', 'Output']

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

w_ML = np.linalg.inv(X.T @ X) @ (X.T @ y)

def sgd(X, y, w_initial, learning_rate, iterations, batch_size):
    w = w_initial
    n = X.shape[0]
    past_history = []

    for _ in range(iterations):

        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            index = indices[start:end]
            X_batch = X[index]
            y_batch = y[index]

            y_pred = X_batch @ w
            sgd_ = X_batch.T @ (y_pred - y_batch)
            w = w - learning_rate * sgd_

        norm_diff = np.linalg.norm(w - w_ML)**2
        past_history.append(norm_diff)

    return w, past_history

learning_rate = 0.00001
iterations = 500
batch_size = 100
w_initial = np.zeros(X.shape[1])


w_final, past_history = sgd(X, y, w_initial, learning_rate, iterations, batch_size)

plt.figure(figsize=(10, 6))
plt.plot(range(iterations), past_history, label='∥wt − wML∥2')
plt.xlabel('Iteration')
plt.ylabel('Squared Norm of Difference')
plt.title('Convergence of Stochastic Gradient Descent (Batch Size = 100)')
plt.legend()
plt.grid(True)
plt.show()
print("w_final after 500 iteration",w_final)