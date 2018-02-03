'''
Created on Dec 14, 2016
Python 3.5.2
@author: Nidhalios
'''

import numpy as np

import models.multiDTW as multiDTW
from models.load_data import load_data
from models.utils import normalize

"""
def classification(series):
    
    # Mapping table for target classes
    labels = {1:'Changed', -1:'Unchanged'}
    
    # Randomly Sample 70% for Training and 30% for Testing, and Transform them to Tuples
    TRAIN_PERCENTAGE = 70
    t = len(series) * TRAIN_PERCENTAGE // 100
    indicies = random.sample(range(len(series)), t)
    train = [series[i] for i in indicies]
    train_tuples = list(map(lambda x: (x['id'],
                                       int(x['target']),
                                       normalize(np.array(x['series']))),train))
    test = [series[i] for i in range(len(series)) if i not in indicies]
    test_tuples = list(map(lambda x: (x['id'],
                                       int(x['target']),
                                       normalize(np.array(x['series']))),test))
    
    
    # Train a K-nearest neighbor classifier that uses Dynamic Time Warping (DTW) to  
    # evaluate distance between two given Multivariate Time Series 
    KNN_NEIGHBORS = 2
    results = []
    for i in range(len(test_tuples)):
        print('{}/{}...'.format(i+1, len(test_tuples)))
        id,pred,proba = multiDTW.predict(test_tuples[i], train_tuples, KNN_NEIGHBORS)
        results.append([id,pred,proba])
    
    # Display Classification metrics like precision, recall and support to have an idea on 
    # the k-NN model prediction performance 
    tar_results = [row[1] for row in results]
    tar_test = [x[1] for x in test_tuples]
    print(classification_report(tar_results,tar_test, target_names=[l for l in labels.values()]))
    conf_mat = confusion_matrix(tar_results,tar_test)

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
"""


def classification(series):
    data = list(map(lambda x: (x['id'], int(x['target']), normalize(np.array(x['series']))), series))
    multiDTW.cross_validation(data, 5)


def main():
    series = load_data()
    classification(series)
    
    
if __name__ == '__main__': main()