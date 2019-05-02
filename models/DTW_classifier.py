"""
Created on Dec 15, 2016
Python 3.5.2
@author: Nidhalios
My Modifications:
    - upgraded to python 3
    - adapted LB_Keogh and DTWDistance calculation to multivariate time series (http://ciir.cs.umass.edu/pubfiles/mm-40.pdf)
    - evolved the classifier to K-NN instead of 1-NN (with probability output)
Source: https://github.com/alexminnaar/time-series-classification-and-clustering.git
"""

import operator

import numpy as np

from models.utils import column


class DTW_classifier(object):
    @staticmethod
    def DTW_distance(ts1, ts2, w=None):
        """
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        """
        DTW = {}

        if w:
            w = max(w, abs(len(ts1) - len(ts2)))

            for i in range(-1, len(ts1)):
                for j in range(-1, len(ts2)):
                    DTW[(i, j)] = float('inf')

        else:
            for i in range(len(ts1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(ts2)):
                DTW[(-1, i)] = float('inf')

        DTW[(-1, -1)] = 0

        for i in range(len(ts1)):
            if w:
                for j in range(max(0, i - w), min(len(ts2), i + w)):
                    dist = 0
                    l = 1 if not isinstance(ts1[0], list) else len(ts1[0])
                    for x in range(l):
                        dist += (column(ts1, x)[i] - column(ts2, x)[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
            else:
                for j in range(len(ts2)):
                    dist = 0
                    l = 1 if not isinstance(ts1[0], list) else len(ts1[0])
                    for x in range(l):
                        dist += (column(ts1, x)[i] - column(ts2, x)[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

        return np.sqrt(DTW[len(ts1) - 1, len(ts2) - 1])

    @staticmethod
    def LB_keogh(ts1, ts2, r):
        """
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        """
        n1 = len(ts1)
        n2 = len(ts2)
        n = min(n1, n2)
        if r >= n:
            r = n - 1
        LB_sum = 0
        l = 1 if not isinstance(ts1[0], list) else n1
        for x in range(l):
            for ind, v in enumerate(column(ts1, x)):
                if ind >= n:
                    break
                j1 = max(ind - r, 0)
                j2 = min(ind + r, n)
                t = column(ts2, x)[j1:j2]
                lower_bound = min(t)
                upper_bound = max(t)
                if v > upper_bound:
                    LB_sum += (v - upper_bound) ** 2
                elif v < lower_bound:
                    LB_sum += (v - lower_bound) ** 2

        return np.sqrt(LB_sum)

    @staticmethod
    def get_classification(scores, neighbors, nb_classes):
        scores = sorted(scores, key=lambda x: x[1])  # sort in ascending order of scores
        scores = scores[:neighbors]  # take the K lowest scores
        res = np.zeros(nb_classes)
        # Modified this to fit the two target labels of this dataset
        for p in scores:
            if p[2] == -1:
                res[0] += 1
            else:
                res[1] += 1
        index, value = max(enumerate(res), key=operator.itemgetter(1))
        KNN_prediction = -1 if index == 0 else 1
        KNN_proba = value / neighbors
        return int(KNN_prediction), round(KNN_proba, 2)

    @staticmethod
    def predict_many(train, test, labels, window=5, neighbors=5, nb_classes=2, progress=False):
        """
        K-nearest neighbor classification algorithm using LB_Keogh lower 
        bound as similarity measure. Option to use DTW distance instead
        but is much slower.
        """
        preds = []
        k = 1
        for ind, i in test.iterrows():
            if progress:
                print('{}/{}...'.format(k, len(test)))
                k += 1
            min_dist = float('inf')
            closest_ts = []
            for ind, j in train.iterrows():
                if DTW_classifier.LB_keogh(i.series, j.series, window) < min_dist:
                    dist = DTW_classifier.DTW_distance(i.series, j.series, window)
                    if dist < min_dist:
                        min_dist = dist
                        closest_ts.append([j.id, dist, j.target])
            KNN_pred, KNN_proba = DTW_classifier.get_classification(closest_ts, neighbors, nb_classes=nb_classes)
            print("Obs Type {} | Target {} ==> {} [{}]".format(i.path_id, i.target, labels[KNN_pred], KNN_proba))
            preds.append([KNN_pred, KNN_proba])
        return preds

    @staticmethod
    def predict_one(train, test, window=5, neighbors=5, nb_classes=2):
        """
        K-nearest neighbor classification algorithm using LB_Keogh lower 
        bound as similarity measure. Option to use DTW distance instead
        but is much slower.
        """
        min_dist = float('inf')
        closest_ts = []
        for ind, j in train.iterrows():
            if DTW_classifier.LB_keogh(test.series, j.series, window) < min_dist:
                dist = DTW_classifier.DTW_distance(test.series, j.series, window)
                if dist < min_dist:
                    min_dist = dist
                    closest_ts.append([j.id, dist, j.target])
        KNN_pred, KNN_proba = DTW_classifier.get_classification(closest_ts, neighbors, nb_classes=nb_classes)
        return [KNN_pred, KNN_proba]
