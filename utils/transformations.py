import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class Transformations(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.amount_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        pass

    def fit(self, X, y=None):
        self.amount_scaler.fit(X["Amount"].values.reshape(-1, 1))
        self.time_scaler.fit(X["Time"].values.reshape(-1, 1))
        return self
    
    def fit_transform(self, X, y = None):
        X = X.copy()
        X["Amount"] = self.amount_scaler.transform(X["Amount"].values.reshape(-1, 1))
        X["Time"] = self.time_scaler.transform(X["Time"].values.reshape(-1, 1))
        return X

