"""
Project: Indoor-User-Movement-Classification
Author: Nidhalios
Created On: 3/23/17
Python 3.5.2
"""
import random

import numpy as np
from fastdtw import fastdtw
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from models.load_data import load_data
from models.utils import column


def fastDTW_distance(x, y):
    """
        Returns the optimal euclidean distance between the two given multivariate
        times series using the fastDTW algorithm
    """
    distance, path = fastdtw(column(x, 1), column(y, 1), dist=euclidean)
    return distance


def classification(series):
    # Mapping table for target classes
    labels = {1: 'Changed', -1: 'Unchanged'}

    # Randomly Sample 70% for Training and 30% for Testing, and Transform them to Tuples
    TRAIN_PERCENTAGE = 70
    t = len(series) * TRAIN_PERCENTAGE // 100
    indicies = random.sample(range(len(series)), t)
    train = [series[i] for i in indicies]
    trainX = [x['series'] for x in train]
    trainY = [int(x['target']) for x in train]
    test = [series[i] for i in range(len(series)) if i not in indicies]
    testX = [x['series'] for x in test]
    testY = [int(x['target']) for x in test]
    print(trainX)
    print(trainY)

    neigh = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', metric=fastDTW_distance)
    neigh.fit(trainX, np.array(trainY).reshape(-1, len(trainY)))
    predY = neigh.predict(testX)

    # Display Classification metrics like precision, recall and support to have an idea on
    # the k-NN model prediction performance
    print(classification_report(predY, testY[1], target_names=[l for l in labels.values()]))
    conf_mat = confusion_matrix(predY, testY[1])

    # Plot the classification confusion matrix
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


def main():
    series = load_data()
    s = series[:10]
    classification(s)


if __name__ == '__main__': main()
