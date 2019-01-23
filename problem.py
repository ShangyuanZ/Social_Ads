import os
import pandas as pd
import numpy as np
import rampwf as rw
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit


problem_title = 'Tencent Social Ads'
_target_column_name = 'label'
_prediction_label_names = [0, 1]
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
workflow = rw.workflows.FeatureExtractorClassifier()


class MAE(ClassifierBaseScoreType):

    def __init__(self, name='mae', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_proba):
        return np.mean(np.abs((y_true - y_proba)))


class F1_Score(ClassifierBaseScoreType):

    def __init__(self, name='f1', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_proba):
        prec = precision_score(y_true, y_proba)
        recall = recall_score(y_true, y_proba)
        return 2 * prec * recall / (prec + recall)


class W_FScore(ClassifierBaseScoreType):

    def __init__(self, name='wf', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_proba):
        return f1_score(y_true, y_proba, average='weighted')


score_types = [rw.score_types.ROCAUC(name="roc_auc"), 
            W_FScore(), F1_Score(), MAE(), 
            rw.score_types.Accuracy(name="acc"), 
            rw.score_types.NegativeLogLikelihood(name="nll")]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=12)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    adfea = pd.read_csv(os.path.join(path, 'data', 'adFeature.csv'))
    X_df = pd.merge(X_df, adfea, on='aid', how='inner')
    del adfea
    ufea = pd.read_csv(os.path.join(path, 'data', 'userFeature.csv'))
    X_df = pd.merge(X_df, ufea, on='uid', how='inner')
    del ufea
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)