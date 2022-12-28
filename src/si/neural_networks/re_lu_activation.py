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

    def backward_relu(self, x: np.ndarray, error: np.ndarray = 1, ) -> np.ndarray:
        # backward da layer
        '''
        Backward error propagation in a ReLU activation layer

        :param x: a given data
        :param error: value error of the loss function
        :param learning_rate: learning rate of the function
        :return: error of the previous layer
        '''

        #substituir valores (no self.x) superiores a 0 por 1
        #substituir valores (no self.x) inferiores a 0 por 0
        #multiplicar elemento a elemento entre o erro e os valores anteriores

        x = np.where(np.array(x) > 0, 1, 0)
        # estrutura np.where(condicao x>0, substituir por 1, else substituir por 0)

        error_to_propagate = error * x

        return error_to_propagate


if __name__ == '__main__':
    import numpy as np

    x = np.array([1, 3, 5, -2, -5, 0])
    print(x)
    model = ReLuActivation()
    y = model.backward_relu(x)
    print(y)

'''
    x = np.array([1, 3, 5, -2, -5, 0])
    print(x)
    y = np.where(x > 0, 1, 0)
    print(y)
    '''