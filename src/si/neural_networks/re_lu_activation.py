# -*- coding: utf-8 -*-

from typing import Callable

import numpy as np


class ReLuActivation:
    '''
    This layer must calculate the rectified linear relationship 
    Consider the positive part of your argument

    The activation function is responsible for transforming the 
    summed weighted input from the node into the activation of 
    the node or output for that input.

    '''

    def __init__(self) -> None:
        pass

    def re_lu_activation(self, x: np.ndarray) -> np.ndarray:
        '''
        ReLU formula: f(x) = max(0,x)
        Only consider the positive part --- begins at 0

        :param x: a given data
        :return re_lu: result of ReLu function
        '''

        re_lu = np.maximum(0, x)
        return re_lu
