import torch
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.01, regularization=0.01, num_iters=1000):
        self.learning_rate = learning_rate  # Step size for weight updates
        self.regularization = regularization  # Regularization parameter for weight penalization
        self.num_iters = num_iters  # Number of iterations for training
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        """Train the SVM model using gradient descent."""
        num_samples, num_features = X.shape
        self.w = torch.randn(num_features, dtype=torch.float64, requires_grad=True)  # Initialize weights
        self.b = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)  # Initialize bias

        # Ensure labels are either -1 or 1
        y_i = torch.where(y <= 0, -1, 1)

        # Training loop
        for _ in range(self.num_iters):
            for idx, x_i in enumerate(X):
                # Condition: y_i * (w*x_i - b) >= 1 (SVM constraint)
                condition = y_i[idx] * (torch.dot(x_i, self.w) - self.b)

                loss = torch.max(torch.tensor(0.0, dtype=torch.float64), 1 - condition)
                loss.backward()
                
                with torch.no_grad():
                    # Gradient update for hinge loss with regularization
                    gradient = 2 * self.regularization * self.w - (y_i[idx] * x_i if condition < 1 else 0)
                    self.w -= self.learning_rate * gradient
                    self.b -= self.learning_rate * (y_i[idx] if condition < 1 else 0)

    def predict(self, X):
        """Predict class labels for samples in X."""
        # Compute the decision boundary: w*X - b
        sign = (X @ self.w) - self.b
        return torch.sign(sign)

    def score(self, X, y):
        """Calculate the accuracy of the model."""
        y_pred = self.predict(X)
        accuracy = torch.sum(y == y_pred).item() / len(y)
        return accuracy

    def visualize(self, X, y):
        """Visualize the decision boundary and margin."""
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        # Get the minimum and maximum points along x-axis for plotting
        x0_1 = torch.amin(X[:, 0]).item()
        x0_2 = torch.amax(X[:, 0]).item()

        # Compute decision boundary and margins
        x1_1 = get_hyperplane_value(x0_1, self.w.detach().numpy(), self.b.item(), 0)
        x1_2 = get_hyperplane_value(x0_2, self.w.detach().numpy(), self.b.item(), 0)

        # Margins (offset by ±1)
        x1_1_m = get_hyperplane_value(x0_1, self.w.detach().numpy(), self.b.item(), -1)
        x1_2_m = get_hyperplane_value(x0_2, self.w.detach().numpy(), self.b.item(), -1)
        x1_1_p = get_hyperplane_value(x0_1, self.w.detach().numpy(), self.b.item(), 1)
        x1_2_p = get_hyperplane_value(x0_2, self.w.detach().numpy(), self.b.item(), 1)

        # Plot decision boundary
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")  # Decision boundary
        # Plot margins
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")  # Negative margin
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")  # Positive margin

        # Set y-limits for better visualization
        x1_min = torch.amin(X[:, 1]).item()
        x1_max = torch.amax(X[:, 1]).item()
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()
