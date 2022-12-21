# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable
import numpy as np
from numpy import log as ln
from src.si.data.dataset import Dataset

#formula do ppt
def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Difference of 2 probabilities, true and predicted values

    :param y_true: The true labels of the dataset
    :param y_pred: The predicted labels of the dataset
    :return: The cross-entropy loss function
    '''
    # menos o sumatorio do y vezes o ln de y a dividir pelo nº de exemplos de y
    cross_entropy_value = - np.sum(y_true) * np.ln(y_pred) / len(y_true)
    return cross_entropy_value


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    Returns the derivative of the cross-entropy loss function

    :param y_true: The true labels of the dataset
    :param y_pred: The predicted labels of the dataset
    :return: The derivative of the cross-entropy loss function
    '''
    # função anterior derivada em ordem a y
    # menos (valor real sobre previsto + 1 menos o valor real a dividir por 1 menos o previsto, tudo a dividir pelo nº de exemplos)

    cross_entropy_derivative_value = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)
    return cross_entropy_derivative_value
