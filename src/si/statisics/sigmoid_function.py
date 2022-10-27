# -*- coding: utf-8 -*-
from typing import Tuple, Union
from si.data.dataset import Dataset
import numpy as np
def sigmoid_function(X):
    '''
        Returns the sigmoid function of a given input

    :param X: input value
    :return: sigmoid function

    '''
    sigmoid = 1 / (1 + np.exp(-X))

    return sigmoid

