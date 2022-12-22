# -*- coding: utf-8 -*-

import numpy as np
from src.si.statisics.sigmoid_function import sigmoid_function

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
        self.weights = np.random.randn(input_size, output_size) * 0.01 #hiperparametro dado
        self.bias = np.zeros((1, output_size)) # serve para evitar o overfitting e inicializa o bias

        self.X = None # para usar no backwasd e forward -- determinada data

    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Performs a forward pass of the layer using the given input
        Returns a 2d numpy array with shape (1, output_size)

        :param X: The input to the layer
        :return output: The output of the layer
        '''
        self.X = X # dado no init
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        '''
        It computes the backward pass of the layer

        :param error: value error of the loss function
        :param learning_rate: learning rate of the function
        :return: error of the previous layer
        '''
        #contabiliza o erro multiplicando o erro pelos pesos
        error_to_propagate = np.dot(error, self.weights.T)
        # Vai atualizando os pesos e muda o bias
        # tem de ser transposta pq trata-se de multiplicação de matrizes (np.dot)
        self.weights = self.weights - learning_rate * np.dot(self.X.T, error) 
        #bias  'e o bias menos a taxa de aprendizagem a multiplicar pela somatorio do erro
        # tem de ser somatorio pq o bias e o erro tem dimensoes diferentes
        # # bias no forward tem sum pq ha varias camadas 
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)  
    
        return error_to_propagate


class SigmoidActivation:
    '''
    A sigmoid activation layer.
    '''
    def __init__(self):
        '''
        Initialize the sigmoid activation layer.
        '''
        self.X = None

    @staticmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Performs a forward pass of the layer using the given input
        Returns a 2d numpy array with shape (1, output_size)

        :param X: The input to the layer
        :return output: The output of the layer
        '''

        self.X = X
        return sigmoid_function(X)

    def backward(self, error: np.ndarray) -> np.ndarray:
        '''
        It performs a backward

        :param error: error
        :param learning_rate: learning rate of a model
        '''
        
        #usar a formula da sigmoid function para o calculo
        sigmoid_derivative = sigmoid_function(self.x) * (1 - sigmoid_function(self.x))
        #get error for previous layer - erro dado
        error_to_propagate = error * sigmoid_derivative

        return error_to_propagate


        