import numpy as np
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        y_pred = np.zeros((len(X), 2))
        # y_pred[:, 0] = np.random.randint(2, size=len(X))
        y_pred[:, 0] = np.random.random(size=len(X))
        y_pred[:, 1] = 1 - y_pred[:, 0]
        return y_pred
