from __future__ import division

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb


class Classifier(BaseEstimator):
    def __init__(self):

        self.model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=31,
            reg_alpha=0.0,
            reg_lambda=1,
            max_depth=-1,
            n_estimators=1500,
            objective='binary',
            subsample=0.7,
            colsample_bytree=0.7,
            subsample_freq=1,
            learning_rate=0.05,
            min_child_weight=50,
            random_state=2018,
            n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
