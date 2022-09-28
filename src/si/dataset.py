# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class Dataset:

    def __init__(self, X,y, features, label):
        '''
            Contructor;
            :parameter
            X : the features matrix/table (independent variables)
            y : dependent variable vector
            features : feature name vector
            label : name of the dependent variable vector
        '''
        self.X = x
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
        return np.mean(self.X, axis = 0)

    def get_variance(self):
        '''
            Returns the variance value of the feature/variable dependent
        '''
        return np.var(self.X, axis = 0)

    def get_median(self):
        '''
            Returns the median value of the feature/variable dependent
        '''
        return np.median(self.X, axis = 0)

    def get_min(self):
        '''
            Returns the minimum value of the feature/variable dependent
        '''
        return np.min(self.X, axis = 0)

    def get_max(self):
        '''
            Returns the maximum value of the feature/variable dependent
        '''
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

