# This tells matplotlib not to try opening a new window for each plot.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def EuclideanDistance(v1, v2):
    sum = 0.0
    for index in range(len(v1)):
        sum += (v1[index] - v2[index]) ** 2
    return sum ** 0.5

class NeighborsClass:
    # Initialize an instance of the class.
    def __init__(self, metric=EuclideanDistance):
        self.metric = metric
    
    # No training for Nearest Neighbors. Just store the data.
    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
    
    # Make predictions for each test example and return results.
    def predict(self, test_data):
        results = []
        for item in test_data:
            results.append(self._predict_item(item))
        return results
    
    # Private function for making a single prediction.
    def _predict_item(self, item):
        best_dist, best_label = 1.0e10, None
        for i in range(len(self.train_data)):
            dist = self.metric(self.train_data[i], item)
            if dist < best_dist:
                best_label = self.train_labels[i]
                best_dist = dist
        return best_label


def train_and_evaluate(train_data, train_labels, test_data, test_labels):
    clf = NeighborsClass()
    print(train_data.shape, train_labels.shape)
    clf.fit(train_data, train_labels)
    preds = clf.predict(test_data)

    correct, total = 0, 0
    for pred, label in zip(preds, test_labels):
        if pred == label: correct += 1
        total += 1

    print ('total: %3d  correct: %3d  accuracy: %3.2f'  %(total, correct, 1.0*correct/total))


def K_train_and_evaluate(train_data, train_labels, test_data, test_labels):
  # Create a KNeighborsClassifier with k=7 (7 neighbors)
    knn_classifier = KNeighborsClassifier(n_neighbors=7)

    # Fit the classifier to the training data
    knn_classifier.fit(train_data, train_labels)

    # Make predictions on the test data
    y_pred = knn_classifier.predict(test_data)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(test_labels, y_pred)
    print("Accuracy:", accuracy)
    return y_pred

    