import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

    params = {
        'depth': np.arange(1, 6),
        'iterations': np.arange(100, 600, 100),
    }
    grid_search = GridSearchCV(estimator=CatBoostClassifier(random_state=42, loss_function='MultiClass'),
                               scoring='accuracy',
                               param_grid=params, 
                               cv=StratifiedKFold(n_splits=5)).fit(X_train_pca, y_train, verbose=False)
    grid_res = pd.DataFrame(grid_search.cv_results_)
    grid_res.to_csv('./grid_res.csv')
