import torch

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_epochs=100, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = torch.randn((num_features, 1), dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        for epoch in range(self.num_epochs):
            # Forward pass
            linear_model = X @ self.weights + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute loss with L2 regularization
            loss = self.loss(y, y_predicted)

            # Backward pass
            loss.backward()

            # Update weights and bias
            with torch.no_grad():
                self.weights -= self.learning_rate * self.weights.grad
                self.bias -= self.learning_rate * self.bias.grad

                # Zero the gradients
                self.weights.grad.zero_()
                self.bias.grad.zero_()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')

    def loss(self, y, y_pred):
        # L2 Regularization term
        reg_term = self.lambda_reg * torch.sum(self.weights ** 2)
        return -torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)) + reg_term

    def predict(self, X):
        linear_model = X @ self.weights + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).float()  # Threshold at 0.5 for classification

    def accuracy(self, y_true, y_pred):
        correct_predictions = (y_true == y_pred).sum().item()
        total_predictions = y_true.shape[0]
        return correct_predictions / total_predictions