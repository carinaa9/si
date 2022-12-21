# -*- coding: utf-8 -*-

import numpy as np


class Dense:
    '''
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.

    :param input_size: The number of inputs the layer will receive
    :param output_size: The number of outputs the layer will produce
    
    Attributes
    ----------
    weights: The weights of the layer
    bias: The bias of the layer
    '''
    def __init__(self, input_size: int, output_size: int):
        '''
        Initialize the dense layer

        :param input_size: The number of inputs the layer will receive
        :param output_size: The number of outputs the layer will produce
        '''
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Performs a forward pass of the layer using the given input
        Returns a 2d numpy array with shape (1, output_size)

        :param X: The input to the layer
        :return output: The output of the layer
        '''

        return np.dot(X, self.weights) + self.bias


class SigmoidActivation:
    '''
    A sigmoid activation layer.
    '''
    def __init__(self):
        '''
        Initialize the sigmoid activation layer.
        '''
        pass

    @staticmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Performs a forward pass of the layer using the given input
        Returns a 2d numpy array with shape (1, output_size)

        :param X: The input to the layer
        :return output: The output of the layer
        '''

        self.X = X
        return 1 / (1 + np.exp(-X))

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        '''
        It performs a backward

        :param error: error
        :param learning_rate: learning rate of a model
        '''

        sigmoid_derivate = 1 / (1 + np.exp(-self.X))
        sigmoid_derivate = sigmoid_derivate * (1 - sigmoid_derivate)

        #get error for previous layer
        error_to_propagate = error * sigmoid_derivate

        return error_to_propagate

