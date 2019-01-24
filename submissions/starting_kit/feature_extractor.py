import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../")
from problem import get_train_data, get_test_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        # X_df_new = X_df[0].copy()
        # data_new = X_df[1].copy()
        X_df_new = X_df.copy()
        train, _ = get_train_data()
        test, _ = get_test_data()
        data_new = pd.concat([train, test])
        
        X_df_new = X_df_new.fillna('-1')  # replace missing values NaN
        data_new = data_new.fillna('-1')

        one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility',
            'education','gender', 'house', 'os', 'ct', 'marriageStatus',
            'advertiserId', 'campaignId', 'creativeId', 'adCategoryId',
            'productId', 'productType']  # features with only one scalar
        
        vector_feature = ['appIdAction', 'appIdInstall', 'interest1',
            'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
            'kw2', 'kw3', 'topic1', 'topic2', 'topic3']  # vector features

        X_df_new = labelEncoder(data_new, X_df_new, one_hot_feature)
        data_new = labelEncoder(data_new, data_new, one_hot_feature)  # normalize features

        X_sparse = OneHot(data_new, X_df_new, one_hot_feature)
        X_sparse = Vectorize(data_new, X_df_new, vector_feature, X_sparse)

        return X_sparse.tocsr()


def labelEncoder(data, X_df, one_hot_feature):  # normalize features
    le = LabelEncoder()
    for feature in one_hot_feature:
        try:
            le.fit(data[feature].apply(int))
            X_df[feature] = le.transform(X_df[feature].apply(int))

        except:
            le.fit(data[feature])
            X_df[feature] = le.transform(X_df[feature])

    return X_df


def OneHot(data, X_df, one_hot_feature):
    enc = OneHotEncoder()
    X_sparse = X_df[['creativeSize']]
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        X_onehot = enc.transform(X_df[feature].values.reshape(-1, 1))
        X_sparse = sparse.hstack((X_sparse, X_onehot))
    print('one hot finished')
    return X_sparse


def Vectorize(data, X_df, vector_feature, X_sparse):
    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        X_vec = cv.transform(X_df[feature])
        X_sparse = sparse.hstack((X_sparse, X_vec))
    print('cv finished')
    return X_sparse
