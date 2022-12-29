# -*- coding: utf-8 -*-

from typing import Callable

import numpy as np

from src.si.data.dataset import Dataset
from src.si.neural_networks.layer import Dense
from src.si.metrics.accuracy import accuracy
from src.si.metrics.mse import mse_derivative, mse



class NN:
    '''
    The NN is the Neural Network model.
    It comprehends the model topology including several neural network layers.
    The algorithm for fitting the model is based on backpropagation.

    :param layers: List of layers in the neural network.
    '''
    #Uma rede neuronal é um processador paralelo capaz de adquirir 
    # conhecimento em problemas não lineares e complexos.

    # Iremos implementar um modelo de rede neuronal genérico 
    # que permite definir arquiteturas (topologias) complexas---NN

    def __init__(self, layers: list, epochs: int = 1000, learning_rate: float = 0.01,
                 loss_function: Callable = mse, loss_derivative : Callable = mse_derivative,
                 verbose: bool = False):
        '''
        Initialize the neural network model.

        :param layers: List of layers in the neural network.
        :param epochs: Number of epochs to train the model
        :param learning_rate: learning rate
        :param loss_function: loss function
        :param loss_derivate: loss derivate function
        :param verbose: if True will print the loss of each epoch
        '''

        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        self.history = {} # guarda os resultados do erro/ custo (valores previstos vs reais) de cada epoch

    def fit(self, dataset: Dataset) -> 'NN':
        '''
        It fits the model to the given dataset.

        :param dataset: The dataset to fit the model to
        :return self: NN. The fitted model
        '''
        # calculamos os custos e alteramos os dos anteriores
        # faz se isto x vezes pelo nº de epochs
        X = dataset.X
        y = dataset.y

        for epoch in range(1, self.epochs + 1):

            # forward propagation
            y_true = np.reshape(dataset.y, (-1, 1))
            y_pred = np.array(dataset.X)

            for layer in self.layers:
                y_pred = layer.forward(y_pred)

            # calcular o custo e fazer backward propagation
            error = self.loss_derivative(y, y_pred)
            for layer in self.layers[::-1]: # ultima layer
                # lista de erro ao contrario
                error = layer.backward(error, self.learning_rate)

        # save history
        cost = self.loss(y, y_pred)
        self.history[epoch] = cost

        #print loss ---> secundario
        if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} --- cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        It predicts the output of the given dataset.

        :param dataset: The dataset to predict the output of
        :return predictions: np.ndarray. The predicted output
        '''
        X = dataset.X

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, dataset: Dataset) -> float:
        '''
        It computes the cost of the model on the given dataset.

        :param dataset: The dataset to compute the cost on
        :return cost: The cost of the model
        '''
        y_pred = self.predict(dataset)
        return self.loss_function(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        '''
        It computes the score of the model on the given dataset.

        :param dataset: The dataset to compute the score on
        :param scoring_func: The scoring function to use
        :return score: The score of the model
        '''
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)


if __name__ == '__main__':
    
    pass