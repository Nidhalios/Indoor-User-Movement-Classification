"""
Created on Dec 16, 2016
Python 3.5.2
@author: Nidhalios
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from models.load_data import load_data
from models.utils import column


def get_features(rss_series):
    """
    Extract basic statistical features from the RSS multivariate time series
    """
    feats = []
    for i in range(4):
        rss = column(rss_series, i)
        feats.append(min(rss))
        feats.append(max(rss))
        feats.append(np.mean(rss))
        feats.append(np.median(rss))
        feats.append(np.var(rss))
        feats.append(np.std(rss))
        feats.append(kurtosis(rss))
        feats.append(skew(rss))
        q25, q75 = np.percentile(rss, q=[25, 75])
        feats.append(q25)
        feats.append(q75)
        feats.append(q75 - q25)

    return feats


def classify(series):
    # Mapping table for target classes
    labels = {1: 'Changed', -1: 'Unchanged'}

    # Stratified (by target class) Shuffle (random) split for train/test data
    targets = series['target'].values.tolist()
    sss = StratifiedShuffleSplit(test_size=0.33, random_state=0)
    train_idx, test_idx = next(sss.split([[0]] * len(targets), targets))
    train_set = series.iloc[train_idx,]
    test_set = series.iloc[test_idx,]

    trainX = train_set['series'].apply(lambda x: get_features(x)).tolist()
    trainY = train_set['target'].values.tolist()
    testX = test_set['series'].apply(lambda x: get_features(x)).tolist()
    testY = test_set['target'].values.tolist()

    # KNN algorithm in a feature based approach this time
    # Let's try also to find the best combination of hyperparameters (number of neighbors, neighbors weights,
    # and distance metric) to get the most out of our model.

    from sklearn.model_selection import GridSearchCV

    params_grid = {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Here I chose the f1 metric as the score to maximize in this gridsearch, as it combines and balances
    # the importance of both recall and precision.
    gridSearch = GridSearchCV(KNeighborsClassifier(), params_grid, cv=3, n_jobs=-1, verbose=1, scoring="f1")
    gridSearch.fit(trainX, trainY)

    print('Best hyperparameters: ', gridSearch.best_params_)
    print('Best score: ', gridSearch.best_score_)

    bestKNN = gridSearch.best_estimator_
    predY = bestKNN.predict(testX)

    # Display Classification metrics like precision, recall and support to have an idea on
    # the k-NN model prediction performance
    print(classification_report(predY, testY, target_names=[l for l in labels.values()]))
    conf_mat = confusion_matrix(predY, testY)

    # Plot the confusion matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(conf_mat), cmap=plt.get_cmap('summer'), interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            plt.text(j - .2, i + .1, c, fontsize=14)

    plt.title('Confusion Matrix')
    _ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
    _ = plt.yticks(range(2), [l for l in labels.values()])
    plt.tight_layout()
    plt.show()

    print('Done.')


def main():
    """Where the magic happens"""
    series = load_data()
    classify(series)


if __name__ == '__main__': main()
