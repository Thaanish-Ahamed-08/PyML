import torch

class KNN:
    def __init__(self, n_neighbors=3, distance_metric='euclidean'):
        self.k = n_neighbors
        self.distance_metric = distance_metric
    
    def fit(self, X, y, loss_history=None):
        # Ensure X and y are tensors
        self.X_train = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.y_train = torch.tensor(y, dtype=torch.int64) if not isinstance(y, torch.Tensor) else y

    def predict(self, X):
        # Ensure X is a tensor
        X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        y_pred = [self._predict(x) for x in X]
        return torch.tensor(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = self._compute_distances(x)
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = torch.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        # Convert k_nearest_labels to integer type before bincount
        k_nearest_labels = k_nearest_labels.to(torch.int64)  # Ensure it's int64

        # Return the most common class label among the k neighbors
        most_common = torch.bincount(k_nearest_labels).argmax()
        return most_common.item()
    
    def _compute_distances(self, x):
        x = x.unsqueeze(0)  # Add a dimension to x for broadcasting

        # Compute distance based on the selected metric
        if self.distance_metric == 'euclidean':
            distances = torch.sqrt(torch.sum((self.X_train - x) ** 2, dim=1))
        elif self.distance_metric == 'manhattan':
            distances = torch.sum(torch.abs(self.X_train - x), dim=1)
        elif self.distance_metric == 'minkowski':
            p = 3  # You can adjust p
            distances = torch.pow(torch.sum(torch.abs(self.X_train - x) ** p, dim=1), 1/p)
        else:
            raise ValueError("Unsupported distance metric.")
        
        return distances
    
    def score(self, X_test, y_test):
        # Predict the labels for the test set
        y_pred = self.predict(X_test)
        
        # Calculate the accuracy
        accuracy = torch.mean((y_pred == y_test).float())
        return accuracy.item()
