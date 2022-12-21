# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Tuple, Union, Callable

from src.si.data.dataset import Dataset
from src.si.statisics.euclidean_distance import euclidean_distance


class KMeans:
    '''
    It performs k-means clustering on the dataset
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid
    It returns the centroids and the indexes of the closest centroid for each point
    '''

    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        '''
        K-means clustering algorithm

        :param k: number of clusters/centroids
        :param max_iter: maximum number of iterations
        :param distance: function that calculates the distance
        '''
        # parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        # attributes
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):

        #Constroi os centroides fora do fit atraves de permutation
        '''
        Infers the centroids by minimizing the distance between the samples and the centroid
        It generates initial k centroids

        :param dataset: a given dataset
        '''
        samples = np.random.permutation(dataset.shape()[0][:self.k])  # cria uma permutation com o tamanho do dataset e seleciona apenas k amostras
        self.centroids = dataset.X[samples, :]

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        #calcula a distancia entre uma amostra e os varios centroides
        #escolhe o centroide com a distancia mais curta e aplica o metodo a todas as amostras do dataset dado

        '''
        Get the closest centroid to each data point

        :param sample : np.ndarray, shape=(n_features,). A sample
        :return np.ndarray: The closest centroid to each data point
        '''
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis=0) # o np.argmin dá o indice da menor distancia // por norma o axis = 0 sao linhas e 1 é colunas

        return closest_centroid_index

    def fit(self, dataset: Dataset) -> 'KMeans':
        #calcula a media de cada centroide, sendo que agrupa as varias amostras pelos seus centroides
        # aplica os objetos centroids e closer_centroids

        '''
        It fits k-means clustering on the dataset
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter
        Convergence is reached when the centroids do not change anymore

        :param dataset: Dataset object
        :return KMeans: KMeans object
        '''
        # generate initial centroids
        self._init_centroids(dataset)  #centroides iniciais

        # fitting the k-means
        convergence = False # para acabar quando convergir os valores da distancia
        i = 0
        labels = np.zeros(dataset.shape()[0])
        while not convergence and i < self.max_iter:

            # get closest centroid
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X) # aplica a funcao a todas as amostras

            # compute the new centroids
            centroids = [] # lista dos centroides
            # juntar as amostras
            for c in range(self.k):

                centroid = dataset.X[self.labels == c]
                centroid = np.mean(centroid, axis=0)  # array de uma só dimensão (varias linhas de um so atributo)
                centroids.append(centroid)

            self.centroids = np.array(centroids)

            # check if the centroids have changed
            convergence = np.any(new_labels != labels) # verificar se mudaram

            # replace labels
            labels = new_labels

            # increment counting
            i += 1 # mudar as labels e contar

        self.labels = labels  # depois de trocar o nome

        return self

    def _get_distances(self, sample: np.ndarray) -> np.ndarray:
        '''
        It computes the distance between each sample and the closest centroid

        :param sample : np.ndarray, shape=(n_features,). A sample
        :return np.ndarray: Distances between each sample and the closest centroid
        '''
        return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset) -> np.ndarray:
        # calcula a distancia entre cada amostras e os diversos centroides
        '''
        It transforms the dataset
        It computes the distance between each sample and the closest centroid

        :param dataset: Dataset object
        :return np.ndarray: Transformed dataset
        '''
        centroids_distances = np.apply_along_axis(self._get_distances, axis=1, arr=dataset.X)
        return centroids_distances

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        '''
        It fits and transforms the dataset

        :param dataset: Dataset object
        :return np.ndarray: Transformed dataset
        '''
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset: Dataset) -> np.ndarray:
        # calcula a distancia entre uma amostra e os varios centroides
        # escolhe o centroide com a distancia mais curta e aplica o metodo ao dataset inteiro

        '''
        It predicts the labels of the dataset

        :param dataset: Dataset object
        :return np.ndarray: Predicted labels
       '''
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        '''
        It fits and predicts the labels of the dataset

        :param dataset: Dataset object
        :return np.ndarray: Predicted labels
        '''
        self.fit(dataset)
        return self.predict(dataset)


if __name__ == '__main__':
    from src.si.data.dataset import Dataset
    dataset_ = Dataset.from_random(100, 5)

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)



