# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Tuple, Union, Callable
from scipy import stats
from src.si.data.dataset import Dataset



class PCA:

    def __init__(self, n_components: int) -> None:
        '''
            It performs the Principal Component Analysis (PCA) on a givrn dataset, using the Singular Value Decomposition method.

        :param n_components: Number of components to be considered and returned from the analysis.
        '''
        self.n_components = n_components

        #parametros estimados
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> 'PCA':
        '''
            It fits the data and stores the mean values of each sample, the principial components and the explained variance.

        :param dataset: Dataset 
        '''

        # centrar os dados (inferir média de amostras e subtrair a media ao dataset)
        self.mean = np.mean(dataset.X, axis=0)
        self.centered = dataset.X - self.mean  

        # SVD
        #Vt transposto
        U, S, Vt = np.linalg.svd(self.centered, full_matrices=False)

        # componentes principais (PC) - primeiros componentes do Vt
        self.components = Vt[:, :self.n_components]  # primeiras colunas do n_components

        # variancia explicada (explained variance): formula EV = S^2/(n-1)
        #n= nº amostras, S dado pelo SVD acima
        # variancia explicada --> primeiros n_components de EV

        #formula do ppt - S do svd
        EV = (S ** 2) / (len(dataset.X) - 1)

        #variancia explicada
        explained_variance = EV[:self.n_components] # primeiros componentes [:componentes]
        return explained_variance


    def transform(self, dataset: Dataset) -> Dataset:
        '''
            Returns the calculated reduced Singular Value Decomposition (SVD)

        :param dataset: Dataset 
        '''

        # centrar os dados
        #subtrair media ao dataset e usar a media do fit
        self.mean = np.mean(dataset.X, axis=0)
        centered = dataset.X - self.mean 

        #calcular X reduced
        # V é a transposta de Vt entao tem de se transpor --> self.components.T
        *rest, Vt = (dataset.X - (np.mean(dataset.X, axis = 0)))

        V = Vt.T
        X_reduced = np.dot(centered, V)

        return Dataset(X_reduced, dataset.y, dataset.features, dataset.label)



    def fit_transform(self, dataset: Dataset) -> None:
        '''
            It fit and transform the dataset

        :param dataset: Dataset 
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
    pca = PCA(n_components=5)
    print(pca.fit_transform(dataset=dataset))