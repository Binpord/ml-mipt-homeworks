import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold

if __name__ == '__main__':
    X_train = np.genfromtxt('data/train/X_train.txt')
    y_train = np.genfromtxt('data/train/y_train.txt')

    X_test = np.genfromtxt('data/test/X_test.txt')
    y_test = np.genfromtxt('data/test/y_test.txt')

    with open('data/activity_labels.txt', 'r') as iofile:
        activity_labels = iofile.readlines()

    activity_labels = [x.replace('\n', '').split(' ') for x in activity_labels]
    activity_labels = dict([(int(x[0]), x[1]) for x in activity_labels])

    data_mean = X_train.mean(axis=0)
    data_std = X_train.std(axis=0)

    X_train = (X_train - data_mean)/data_std
    X_test = (X_test - data_mean)/data_std

    unique_columns = np.genfromtxt('unique_columns.txt', delimiter=',').astype(int)
    X_train_unique = X_train[:, unique_columns]
    X_test_unique = X_test[:, unique_columns]

    pca = PCA(0.99)
    X_train_pca = pca.fit_transform(X_train_unique)
    X_test_pca = pca.transform(X_test_unique)

    lgbm_params = {
        'max_depth': np.append(-1, np.arange(1, 11)),
        'n_estimators': np.append(np.arange(100, 1100, 100), np.arange(1500, 5000, 500)),
    }
    lgbm_search = GridSearchCV(estimator=LGBMClassifier(random_state=42), 
                               param_grid=lgbm_params, 
                               cv=StratifiedKFold(n_splits=5)).fit(X_train_pca, y_train)
    lgbm_results = pd.DataFrame(lgbm_search.cv_results_)
    lgbm_results.to_csv('./lgbm_results.csv')

    catboost_params = {
        'depth': np.arange(1, 11),
        'iterations': np.append(np.arange(100, 1100, 100), np.arange(1500, 5000, 500)),
    }
    catboost_search = GridSearchCV(estimator=CatBoostClassifier(random_state=42), 
                                   param_grid=catboost_params, 
                                   cv=StratifiedKFold(n_splits=5)).fit(X_train_pca, y_train)
    catboost_results = pd.DataFrame(catboost_search.cv_results_)
    catboost_results.to_csv('./catboost_results.csv')
