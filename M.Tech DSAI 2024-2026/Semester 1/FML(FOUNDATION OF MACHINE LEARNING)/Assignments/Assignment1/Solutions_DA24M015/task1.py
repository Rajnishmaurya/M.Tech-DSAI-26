import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
train_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_train.csv",header=None)
test_data=pd.read_csv("/content/drive/My Drive/Colab Notebooks/FMLA1Q1Data_test.csv",header=None)

train_data.columns = ['Feature1', 'Feature2', 'Output']

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')

ax.set_xlabel('Feature 1 (x1)')
ax.set_ylabel('Feature 2 (x2)')
ax.set_zlabel('Target(y) ')
ax.set_title('3D Scatter Plot of the Data')
plt.show()

w_ML = np.linalg.inv(X.T @ X) @ (X.T @ y)
print()
print("Least squares solution wML:", w_ML)
print()

def predict(X, w):
    return X @ w

y_pred = predict(X, w_ML)

sse = np.sum((y - y_pred) ** 2)
print("Sum of Squared Errors (SSE):", sse)
print()

fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label='Actual data')

x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1_range, x2_range)

X_surf = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
y_surf = predict(X_surf, w_ML).reshape(x1.shape)

ax.plot_surface(x1, x2, y_surf, color='b', alpha=0.5)

ax.set_xlabel('Feature 1 (x1)')
ax.set_ylabel('Feature 2 (x2)')
ax.set_zlabel('Target (y)')
ax.set_title('3D Plot with Regression Plane')

plt.legend()
plt.show()
