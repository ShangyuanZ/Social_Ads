import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

_target_column_name = 'label'


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
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