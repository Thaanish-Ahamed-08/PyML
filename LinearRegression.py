import torch

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y,loss_history=None):
        num_samples, num_features = X.shape
        # Initialize weights and bias
        self.weights = torch.randn((num_features, 1), dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        for epoch in range(self.num_epochs):
            # Forward pass: Calculate predictions
            y_predicted = X @ self.weights + self.bias

            # Calculate loss (Mean Squared Error)
            loss = self.loss(y, y_predicted)
            if loss_history is not None:
                loss_history.append(loss.item())
            # Backward pass: Compute gradients
            loss.backward()

            # Update weights and bias
            with torch.no_grad():
                self.weights -= self.learning_rate * self.weights.grad
                self.bias -= self.learning_rate * self.bias.grad

                # Zero the gradients after updating
                self.weights.grad.zero_()
                self.bias.grad.zero_()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')

    def loss(self, y, y_pred):
        return torch.mean((y - y_pred) ** 2)  # Mean Squared Error

    def predict(self, X):
        return X @ self.weights + self.bias  # Return predictions

    def mse(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)  # Mean Squared Error
