# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Union
from typing import Callable
from src.si.statisics.euclidean_distance import euclidean_distance
from src.si.data.dataset import Dataset
from src.si.metrics.rmse import rmse


class KNNRegressor:

    def __init__(self, k: int, distance: Callable = euclidean_distance) -> None:
        '''
        This algorithm predicts the class for a sample using the k most similar examples

        :param k: number of examples to consider
        :param distance: euclidean distance function. Defaults to euclidean_distance
        '''
        self.k = k
        self.distance = distance
        #estimados
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        '''
        It stores training dataset

        :param dataset: training dataset
        :return: self
        '''
        #ARMAZENA O DATASET DE TREINO
        self.dataset = dataset  # dataset de treino para o modelo
        return self

    def _get_closest_label(self, sample: np.ndarray):
        '''
        It returns the closest label of the given sample

        :param sample: The sample to get the closest label of
        :return label: The closest label
        '''
        #distancia entre cada amostra e as varias amostras do dataset
        distances = self.distance(sample, self.dataset.X)

        #obtem os indx dos k exemp mais semelhantes(menor distancia)
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        #usa os indx para obter os valores correspondentes em Y
        k_nearest_neighbors_label = self.dataset.y[k_nearest_neighbors] # np.array com as varias classes
        #calcula a media
        return np.mean(k_nearest_neighbors_label)
        

    def predict(self, dataset: Dataset):
        '''
        Calculates the distance between each sample and the various samples in the training dataset.
        Gets the indexes of the k most similar examples (smallest distance).
        Use the previous indexes to get the corresponding Y classes.
        Get the most common class (with the highest frequency) in the k examples

        :param dataset: test dataset
        :return: an array of estimated classes for the test dataset
        '''
        # ESTIMA A CLASSE PARA 1 AMOSTRA TENDO COMO BASSE OS K EXEMP + SEMELHANTES

        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        '''
        Get predictions and calculate the rmse between actual values and forecasts

        :param dataset: test dataset
        :return: calculating the error between forecasts and actual values
        '''
        #CALCULA O ERRO ENTRE AS CLASSES ESTIMADAS E REAIS
        #ERRO: RMSE
        prediction = self.predict(dataset)
        return rmse(dataset.y, prediction)
