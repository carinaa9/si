# -*- coding: utf-8 -*-
from typing import Tuple, Union

import numpy as np
def sigmoid_function(X):
    '''
    Returns the sigmoid function of a given input

    :param X: input value
    :return: sigmoid function
    '''

    sigmoid_value = 1 / (1 + np.exp(-X))
    return sigmoid_value

