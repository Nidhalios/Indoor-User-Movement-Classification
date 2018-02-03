"""
Project: Indoor-User-Movement-Classification
Author: Nidhalios
Created On: 2/3/18
"""
import numpy as np


def normalize(data):
    """
    Normalize the array to avoid outliers and stabilize the values
    """
    for dimension in range(len(data[0])):
        dev = np.std(data[:, dimension])
        mean = np.mean(data[:, dimension])
        for i in range(len(data)):
            data[i][dimension] = (data[i][dimension] - mean) / dev
    return data


def column(matrix, i):
    """
       Returns the column i of the given matrix
    """
    return [row[i] for row in matrix]
