# -*- coding: utf-8 -*-
from typing import Tuple, Union
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    It returns the mean squared error of the model on the given dataset

    :param y_true: The true labels of the dataset
    :param y_pred: The predicted labels of the dataset
    :return mse: The mean squared error of the model
    '''
    mse_value = np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2) # formula ppt5-slide 4
    return mse_value

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    It computes the derivative of mse function

    :param y_true: The true labels of the dataset
    :param y_pred: The predicted labels of the dataset
    :return: The derivative of the MSE function.
    '''

    # formula do ppt
    mse_derivative_value = -2 * (y_true - y_pred) / (len(y_true) * 2)
    return mse_derivative_value

