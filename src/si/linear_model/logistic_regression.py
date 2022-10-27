# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable
import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statisics.sigmoid_function import sigmoid_function

class LogisticRegression:

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, sigmoid_function: Callable = sigmoid_function):
        '''

        :param l2_penalty: The L2 regularization parameter
        :param alpha: The learning rate
        :param max_iter: The maximum number of iterations
        '''
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes - parametros estimados
        self.theta = None # features
        self.theta_zero = None  # é o b da funcao y = mx +b

    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        '''
            Fit the model to the dataset

            :param dataset: The dataset to fit the model to

            :return self: The fitted model
        '''
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)  # o tamanho do theta é o tamanho de colunas/features do dataset
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):  # tem un+m nº max de iterações
            # predicted y
            # y é calculado na mesma como o ridge mas aplica-se a funçãi sigmoid
            y_pred = (np.dot(dataset.X, self.theta)) + self.theta_zero
            y_pred = sigmoid_function(y_pred)


            # computing and updating the gradient with the learning rate
            # calcula o gradiente num dado ponto, subtrai ao gradiente novo o gradiente anterior para descer sempre
            # quanto menor a função de custo, mais perto estamos do minimo
            # gradiente: se o valor do declive for negativo, continua a descer
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)
            # np.dot faz multiplicao entre matrizes

            # computing the penalty
            # termo de penalizacao a dividir pelo nº de amostras a multiplicado pelo theta e multiplicado pelo alpha
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        '''
        Predict the output of the dataset

        :param dataset: The dataset to predict the output of

        :return predictions: The predictions of the dataset
        '''
        # estima os valores de y usando o theta, theta_zero e a função sigmoid
        # converte os valores estimados em 0 e 1
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        mask = predictions >= 0.5
        predictions[mask] = 1
        predictions[~mask] = 0

        return predictions

    def score(self, dataset: Dataset, accuracy: Callable = accuracy) -> float:
        '''
        Compute the accuracy of the model on the dataset

        :param dataset: The dataset to compute the accuracy on

        :return mse: The accuracy of the model
        '''
        # obtem as previsoes usando o predict
        # calcula a accuracyentre os valores reais e as previsoes

        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:  # permite saber quao perto está
        '''
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        :param dataset: The dataset to compute the cost function on

        :return cost: The cost function of the model
        '''
        # permite saber se o nosso gradient decent obteve custos minimoos entre os valores estimados e reais
        # permite ver se o gradiente está a funcionar bem
        # fórmula  no ppt 5 slide 8
        # multiplica o termo de penalização à soma de todos dos thetas ao quadrado dividida pelo dobro do nº das amostras
        # nao é utilizado no gradient (por norma devia estar)

        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1-dataset.y) * np.log(1-predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta **2)/ (2 * dataset.shape()[0]))
        return cost


if __name__ == '__main__':

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = [0, 1] # tem de ser valor entre 0 e 1
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = LogisticRegression()
    model.fit(dataset_)

    # get coefs
    print(f'Parameters (theta): {model.theta}')

    # compute the score
    score = model.score(dataset_)
    print(f'Score (score of the model): {score}')

    # compute the cost
    cost = model.cost(dataset_)
    print(f'Cost: {cost}')

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f'Predictions (y value): {y_pred_}')
