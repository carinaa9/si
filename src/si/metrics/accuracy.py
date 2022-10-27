# -*- coding: utf-8 -*-


import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
        Calculates the error between arguments given using the accuracy formula:
        (VN + VP) / (VN + VP + FP + FN)

    :param y_true: real values.
    :param y_pred: predicted values.

    :return float: Error value between y_true and y_pred
    '''

    return np.sum(y_true == y_pred) / len(y_true)