"""
Created on Dec 14, 2016
Python 3.5.2
@author: Nidhalios
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from models.DTW_classifier import DTW_classifier
from models.load_data import load_data


def classify(series):
    
    # Mapping table for target classes
    labels = {1: 'Changed', -1: 'Unchanged'}

    # Stratified (by target class) Shuffle (random) sampler for train/test data
    targets = series['target'].values.tolist()
    sss = StratifiedShuffleSplit(test_size=0.10, random_state=0)
    train_idx, test_idx = next(sss.split([[0]] * len(targets), targets))
    train_set = series.iloc[train_idx,]
    test_set = series.iloc[test_idx,]
    # Try normalizing the data normalize(np.array(x['series'])))

    classifier = DTW_classifier()
    # Train a K-nearest neighbor classifier that uses Dynamic Time Warping (DTW) to  
    # evaluate distance between two given Multivariate Time Series 
    KNN_NEIGHBORS = 2
    results = classifier.predict_many(train_set, test_set, labels, window=5, neighbors=KNN_NEIGHBORS,
                                      nb_classes=2, progress=True)

    """
    results = []
    for i, item in test_set.iterrows():
        print('{}/{}...'.format(i+1, len(test_set)))
        pred, proba = classifier.predict_one(train_set, item, window=5, neighbors=KNN_NEIGHBORS, nb_classes=2)
        results.append([item.id, pred, proba])
    """
    
    # Display Classification metrics like precision, recall and support to have an idea on 
    # the k-NN model performance
    tar_results = [row[0] for row in results]
    tar_test = test_set.target.values.tolist()
    print([l for l in labels.values()])
    print(classification_report(tar_results, tar_test, target_names=[l for l in labels.values()]))
    conf_mat = confusion_matrix(tar_results, tar_test)

    # Plot the classification confusion matrix 
    plt.figure(figsize=(5,5))
    plt.imshow(np.array(conf_mat), cmap=plt.get_cmap('summer'), interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            plt.text(j-.2, i+.1, c, fontsize=14)

    plt.title('Confusion Matrix')
    _ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
    _ = plt.yticks(range(2), [l for l in labels.values()])
    plt.tight_layout()
    plt.show()

    print('Done.')


def main():
    series = load_data()
    classify(series)
    
    
if __name__ == '__main__': main()