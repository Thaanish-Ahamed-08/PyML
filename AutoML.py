import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

# Assuming your custom PyTorch model implementations
from DecisionTreeClassifier import DecisionTree
from KNN import KNN
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from SVM import SVM

class ModelSelector:
    def __init__(self):
        self.models = {
            'DecisionTree': DecisionTree(),
            'KNN': KNN(),
            'LinearRegression': LinearRegression(),
            'LogisticRegression': LogisticRegression(),
            'SVM': SVM(),
        }

    def get_model(self, model_name):
        return self.models.get(model_name)

class PreprocessingPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return X

def train_and_evaluate(model, X_train, y_train, X_test, y_test, metric):
    loss_history = []
    model.fit(X_train, y_train, loss_history=loss_history)

    predictions = model.predict(X_test)
    predictions = predictions.detach().numpy()

    if predictions.ndim > 1 or predictions.dtype != int:
        predictions = (predictions >= 0.5).astype(int)

    y_test = y_test.detach().numpy()
    
    return metric(y_test, predictions), loss_history

def evaluate_model(predictions, ground_truth, metrics):
    results = {}
    for metric in metrics:
        results[metric.__name__] = metric(ground_truth, predictions)
    return results

def auto_ml_pipeline(X_train, y_train, X_test, y_test, metric):
    best_model = None
    best_score = float('-inf')
    model_selector = ModelSelector()
    
    loss_histories = {}  # Dictionary to store loss histories for each model
    
    for model_name in model_selector.models.keys():
        model = model_selector.get_model(model_name)
        score, loss_history = train_and_evaluate(model, X_train, y_train, X_test, y_test, metric)
        loss_histories[model_name] = loss_history  # Store loss history
        
        print(f"Model: {model_name}, Score: {score}")
        
        if score > best_score:
            best_model = model
            best_score = score
    
    # Plot loss curves after evaluating all models
    plot_loss_curves(loss_histories)

    return best_model, best_score

def plot_loss_curves(loss_histories):
    num_models = len(loss_histories)
    fig, axes = plt.subplots(nrows=num_models, figsize=(12, 5 * num_models), sharex=True)

    for ax, (model_name, loss_history) in zip(axes, loss_histories.items()):
        ax.plot(loss_history, label=model_name)
        ax.set_title(f"Loss Curve for {model_name}")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend()

    plt.tight_layout()
    plt.show()
