import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement


dataFrame = pd.read_parquet("test_pendulum.parquet")
print("DataFrame loaded successfully!")
print("Columns:", dataFrame.columns)
print(dataFrame.head())

SIM_START = 100    # change to 0, 100, 200, etc.
SIM_END = 199    # change to 100, 200, 300, etc.

X = dataFrame[["theta1", "theta2", "theta1_dot", "theta2_dot"]].iloc[:-1].values
y = dataFrame["theta1"].iloc[1:].values

X = X[SIM_START:SIM_END]
y = y[SIM_START:SIM_END]

train_split = len(X) // 2
x_train_data = X[:train_split]
x_test_data = X[train_split:]
y_train_data = y[:train_split]
y_test_data = y[train_split:]

#Normalizing the data

x_mean = np.mean(x_train_data, axis=0)
x_std = np.std(x_train_data, axis=0)
x_train_data = (x_train_data - x_mean) / x_std
x_test_data = (x_test_data - x_mean) / x_std


print("x_train_data shape:", x_train_data.shape)
print("y_train_data shape:", y_train_data.shape)

#X (inputs): theta1, theta2, theta1_dot, theta2_dot
#y (output): theta1_next



# linear regression model (numpy implementation)
class MultipleRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, degree=1):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.degree = degree # 1 for linear regression # degree of polynomial for all features
    def _polynomial_features(self, X):
        poly_features = [X]
        for d in range(2, self.degree + 1):
            poly_features.append(X ** d)
        return np.hstack(poly_features)
    def fit(self, X, y):
        if self.degree > 1:
            X = self._polynomial_features(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = X @ self.weights + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):
        if self.degree > 1:
            X = self._polynomial_features(X)
        return X @ self.weights + self.bias


model = MultipleRegression()
model.fit(x_train_data, y_train_data)
y_predicted = model.predict(x_test_data)

# Calculate Mean Squared Error
mse = np.mean((y_test_data - y_predicted) ** 2)
print("Mean Squared Error:", mse)

t = dataFrame["t"].values
t_sim = t[SIM_START:SIM_END]
t_test = t_sim[train_split:]

plt.plot(t_sim, dataFrame["theta1"].iloc[SIM_START:SIM_END], label="Actual")
plt.plot(t_test, y_predicted, label="Predicted")
plt.xlabel("Time (s)")
plt.ylabel("Theta1 (rad)")
plt.title("Actual vs Predicted Theta1")
plt.legend()
plt.show()
