import torch

class KNN:
    def __init__(self, n_neighbors=3, distance_metric='euclidean'):
        self.k = n_neighbors
        self.distance_metric = distance_metric
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return torch.tensor(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = self._compute_distances(x)
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = torch.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Return the most common class label among the k neighbors
        most_common = torch.bincount(k_nearest_labels).argmax()
        return most_common.item()
    
    def _compute_distances(self, x):
        x = x.unsqueeze(0)  # Add a dimension to x for broadcasting
        
        if self.distance_metric == 'euclidean':
            # Compute the Euclidean distance
            distances = torch.sqrt(torch.sum((self.X_train - x) ** 2, dim=1))
        elif self.distance_metric == 'manhattan':
            # Compute the Manhattan distance
            distances = torch.sum(torch.abs(self.X_train - x), dim=1)
        elif self.distance_metric == 'minkowski':
            # Compute the Minkowski distance with p=3 (you can adjust p)
            p = 3
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
