# -*- coding: utf-8 -*-
import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.mse import mse


class RidgeRegression:
    '''
        The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    '''
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        '''

        :param l2_penalty: he L2 regularization parameter
        :param alpha: The learning rate
        :param max_iter: The maximum number of iterations
        '''
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes - parametros estimados
        self.theta = None
        self.theta_zero = None # é o b da funcao y = mx +b

    def fit(self, dataset: Dataset) -> 'RidgeRegression':
        '''
        Fit the model to the dataset

        :param dataset: The dataset to fit the model to

        :return self: The fitted model
        '''
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n) # o tamanho do theta é o tamanho de colunas/features do dataset
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter): # tem un+m nº max de iterações
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            # calcula o gradiente num dado ponto, subtrai ao gradiente novo o gradiente anterior para descer sempre
            # quanto menor a função de custo, mais perto estamos do minimo
            # gradiente: se o valor do declive for negativo, continua a descer
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)
            #np.dot faz multiplicao entre matrizes

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
        # estima os valores de y usando o theta e theta_zero
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        '''
        Compute the Mean Square Error of the model on the dataset

        :param dataset: The dataset to compute the MSE on

        :return mse: The Mean Square Error of the model
        '''
        # estima dos valores de y usando o theta e o theta_zero
        # calcula o mse entre os valores reais e as previsoes
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float: # permite saber quao perto está
        '''
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        :param dataset: The dataset to compute the cost function on

        :return cost: The cost function of the model
        '''
        # permite saber se o nosso gradient descent obteve custos minimos entre os valores estimados e reais
        # permite ver se o gradiente está a funcionar bem
        # fórmula  no ppt 5 slide 8
        # multiplica o termo de penalização à soma de todos dos thetas ao quadrado dividida pelo dobro do nº das amostras
        # nao é utilizado no gradient (por norma devia estar)

        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))


if __name__ == '__main__':

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegression()
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
