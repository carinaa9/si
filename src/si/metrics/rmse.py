# -*- coding: utf-8 -*-

import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    It calculates the Root Mean Squared Error metric
    :param y_true: Real values
    :param y_pred: Predicted values

    :return float: RMSE between real and predicted values
    '''
    return np.sqrt(((y_true - y_pred) ** 2).mean())