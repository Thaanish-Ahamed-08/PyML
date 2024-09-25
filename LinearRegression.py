import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def loss(self, y_pred, y_actual):
        return np.mean(0.5 * (y_pred - y_actual) ** 2)

    def fit(self, X, y, num_iter = 1000):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1)  # Initialize weights
        self.bias = 0.0  # Initialize bias

        for i in range(num_iter):
            # Compute predictions
            predicted = np.sum((X @ self.weights),self.bias)
            
            # Compute gradients
            dw = (2 / n_samples) * X.T @ (predicted - y)
            db = (2 / n_samples) * np.sum((predicted - y))
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and print loss
            loss = self.loss(predicted, y)
            print(f"Loss: {loss} for epoch {i}")

        return self.weights, self.bias

    def predict(self, X):
        return (X @ self.weights) + self.bias