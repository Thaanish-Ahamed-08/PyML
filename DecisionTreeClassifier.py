import numpy as np

class Node:
    def __init__(self, data=None, children=None, split_on=None, pred_class=None, is_leaf=False):
        self.data = data
        self.children = children
        self.split_on = split_on
        self.pred_class = pred_class
        self.is_leaf = is_leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.root = Node()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def get_params(self, deep=True):
        # Return parameters as a dictionary
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split
        }

    def set_params(self, **params):
        # Set parameters from the dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def calculate_entropy(self, labels):
        entropy = 0
        label_counts = np.unique(labels, return_counts=True)[1]
        total_labels = len(labels)
        for count in label_counts:
            prob = count / total_labels
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy

    def calculate_gini(self, labels):
        label_counts = np.unique(labels, return_counts=True)[1]
        total_labels = len(labels)
        gini = 1 - sum((count / total_labels) ** 2 for count in label_counts)
        return gini

    def get_y(self, data):
        return data[:, -1]

    @staticmethod
    def get_pred_class(Y):
        labels, labels_counts = np.unique(Y, return_counts=True)
        index = np.argmax(labels_counts)
        return labels[index]

    def meet_criteria(self, node, depth):
        y = self.get_y(node.data)
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if self.calculate_entropy(y) == 0:
            return True
        return False
    
    def make_split(self, data, feat_index):
        weighted_entropy = 0
        split_nodes = {}
        total_values = len(data)
        unique_values = np.unique(data[:, feat_index])
        for unique_value in unique_values:
            partition = data[data[:, feat_index] == unique_value, :]
            y = self.get_y(partition)
            node = Node(partition)
            split_nodes[unique_value] = node
            entropy = self.calculate_entropy(y)
            weighted_entropy += (len(partition) / total_values) * entropy

        return split_nodes, weighted_entropy

    def best_split(self, node, depth=0):
        if self.meet_criteria(node, depth):
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return

        split_feature_index = -1
        min_weighted_entropy = float('inf')
        child_nodes = None
        _, num_features = node.data.shape

        for idx in range(num_features - 1):
            split_nodes, weighted_entropy = self.make_split(node.data, idx)
            if weighted_entropy < min_weighted_entropy:
                min_weighted_entropy = weighted_entropy
                split_feature_index = idx
                child_nodes = split_nodes

        if split_feature_index == -1:
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return

        node.children = child_nodes
        node.split_on = split_feature_index

        for child_node in child_nodes.values():
            self.best_split(child_node, depth + 1)

    def traverse_tree(self, x, node):
        if node.is_leaf:
            return node.pred_class
        feat_value = x[node.split_on]
        if feat_value not in node.children:
            return self.get_pred_class(self.get_y(node.data))
        predicted_class = self.traverse_tree(x, node.children[feat_value])
        return predicted_class

    def fit(self, X, y):
        data = np.column_stack([X, y])
        self.root = Node(data)
        self.best_split(self.root)

    def predict(self, X):
        predictions = np.array([self.traverse_tree(x, self.root) for x in X])
        return predictions

    def score(self, y_pred, y):
        return np.mean(y_pred == y)
