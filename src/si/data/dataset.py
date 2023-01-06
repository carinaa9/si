# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple, Sequence


class Dataset:

    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
        '''
        :param X : the features matrix/table (independent variables)
        :param y : dependent variable vector
        :param features : feature name vector
        :param label : name of the dependent variable vector
        '''
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        '''
            Returns dataset dimensions
        '''
        return self.X.shape

    def has_label(self):
        '''
            Checks if the dataset has y (dependent variable)
        '''
        if self.y is not None:
            return True
        return False


    def get_classes(self):
        '''
            Returns the dataset classes (possible y values)
        '''
        if self.y is None:
            raise ValueError('Error')

        return np.unique(self.y)

    def get_mean(self): # se não colocar axis faz a média da matriz toda
        '''
            Returns the mean value of the feature/variable dependent
        '''
        if self.X is None: # se o dataset nao for nulo
            return
        return np.mean(self.X, axis = 0)

    def get_variance(self):
        '''
            Returns the variance value of the feature/variable dependent
        '''
        if self.X is None:
            return
        return np.var(self.X, axis = 0)

    def get_median(self):
        '''
            Returns the median value of the feature/variable dependent
        '''
        if self.X is None:
            return
        return np.median(self.X, axis = 0)

    def get_min(self):
        '''
            Returns the minimum value of the feature/variable dependent
        '''
        if self.X is None:
            return
        return np.min(self.X, axis = 0)

    def get_max(self):
        '''
            Returns the maximum value of the feature/variable dependent
        '''
        if self.X is None:
            return
        return np.max(self.X, axis = 0)

    def summary(self):
        '''
            Returns a pandas DataFrame with all descriptive metrics
        '''
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'variance': self.get_variance(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max()
             }
        )

    def print_dataframe(self):
        '''
            Returns a pandas DataFrame
        '''
        return pd.DataFrame(self.X, self.y, self.features)

    def dropna(self): #Nota que o objeto resultante não deve conter valores nulos em para nenhuma
                    # feature/variável independente. Nota também que deves atualizar o vetor y
                    # removendo as entradas associadas às amostras a remover.
        '''
            Removes all samples that contain at least one null value
        '''

        # cria uma máscara com os registos a manter (com todos os valores preenchidos)
        #np.logical_not computa qualquer valor de self.X que seja NaN e mete como uma mascara
        mask_na = np.logical_not(np.any(np.isnan(self.X), axis=1))


        # filtramos para manter apenas os registos que obedeçam à mascara de cima
        self.X = self.X[mask_na, :]
        self.y = self.y[mask_na]

    # fazer x== NaN
    # new_x = x[mask]
    # nao usar drop na do pandas, tem de ser com o numpy --> logical not

    def fillna(self, value: int): #Nota que o objeto resultante não deve conter valores nulos em para nenhuma feature/variável independente.
        '''
            Replaces all null values by another value that is given

        :param value: a given value
        '''

        self.X = np.nan_to_num(self.X, nan = value)

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])  # matriz
    y = np.array([7, 8])  # vetor
    z = None
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features=features, label=label)
    print('shape:', dataset.shape())
    print('has label:', dataset.has_label())
    print('classes:', dataset.get_classes())
    print('mean:', dataset.get_mean())
    print('variance:', dataset.get_variance())
    print('median:', dataset.get_median())
    print('minimo:', dataset.get_min())
    print('summary:', dataset.summary())
    print('Print of dataset', dataset.print_dataframe())
