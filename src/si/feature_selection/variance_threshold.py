# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from src.si.data.dataset import Dataset


class VarianceThreshold():

    def __init__(self, threshold): # linha de corte ou valor de corte/ tem um atributo
        self.threshold = threshold
        self.variance = None # para ja esta vazio pq nao vamos calcular a variancia no init/ nao temos um dataset aqui

    def fit(self, dataset:Dataset)-> Dataset: # estima o atributo a partir dos dados
        #variance = np.var(dataset.X)
        '''
            Estimates the F and p for each feature using the scoring_func

        :param dataset: a given dataset
        :return: self
        '''
        variance = Dataset.get_variance(dataset)
        self.variance = variance

        return self # retorna se a ele proprio, um api do transform

    def transform(self, dataset:Dataset)-> Dataset:
        '''
            Selects the features with the highest F value up to the indicated percentile.
            (for a dataset with 10 features and a 50% percentile, the transform should select
            the 5 features with higher F value)

        :param dataset: a given dataset
        :return: dataset
        '''

        mask_feat = self.variance > self.threshold
        new_x = dataset.X[:, mask_feat]
        features = np.array(dataset.features)[mask_feat]
        return Dataset(X=dataset.X, y=dataset.y, features=list(features), label = dataset.label)

    def fit_transform(self, dataset:Dataset)-> Dataset:
        '''
            Runs the fit and then the transform

        :param dataset: a given dataset
        :return: transformed dataset
        '''
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':

    dataset = Dataset(X=np.array([[0, 1, 2, 3],
                                  [0, 2, 4, 6],
                                  [1, 3, 5, 7]]),
                      y=np.array([0, 1, 2]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold(threshold=0.1)
    selector = selector.fit(dataset)
    new_dataset = selector.transform(dataset)
    print(dataset.features)