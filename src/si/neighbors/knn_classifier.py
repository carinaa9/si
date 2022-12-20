# -*- coding: utf-8 -*-

import numpy as np

from typing import Callable, Union
from src.si.statisics.euclidean_distance import euclidean_distance
from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy


class KNNClassifier:

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance) -> None:
        '''
            This algorithm predicts the class for a sample using the k most similar examples.
        :param k: number of examples to consider
        :param distance: euclidean distance function. Defaults to euclidean_distance.
        '''
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
        '''
            It stores training dataset
        :param dataset: training dataset

        :return: self
        '''
        self.dataset = dataset  # dataset de treino para o modelo
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        '''
            It returns the closest label of the given sample
        :param sample: The sample to get the closest label of

        :return label: The closest label
        '''
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)
        # get the k nearest neighbors
        k_nearest_neighbor = np.argsort(distances)[:self.k]
        # get the labels of the k nearest neighbors
        k_nearest_neighbor_label = self.dataset.y[k_nearest_neighbor]
        # get the most common label
        labels, counts = np.unique(k_nearest_neighbor_label, return_counts=True)
        
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
            Calculates the distance between each sample and the various samples in the training dataset.
            Gets the indexes of the k most similar examples (smallest distance).
            Use the previous indexes to get the corresponding Y classes.
            Get the most common class (with the highest frequency) in the k examples
        :param dataset: test dataset
        :return: an array of estimated classes for the test dataset
        '''
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        '''
            Get predictions and calculate accuracy between actual values and predictions

        :param dataset: test dataset
        :return: calculating the error between forecasts and actual values
        '''
        prediction = self.predict(dataset)
        return accuracy(dataset.y, prediction)