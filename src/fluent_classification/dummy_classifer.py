from random import random


class DummyClassifier:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def predict(self):
        return random() > self.threshold

    def predict_proba(self, X):
        return self.threshold
