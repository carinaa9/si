import numpy as np
import pandas as pd

class Dataset:

    def __init__(self, X,y, features, label):
        self.X = x
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        return self.X.shape

    def has_label(self):
        if self.y is not None:
            return True
        return False

    def get_classes(self):
        if self.y is None:
            return                             ## colocar :   raise ValueError

        return np.unique(self.y)

    def get_mean(self):
        return np.mean(self.X, axis = 0)

    def get_variance(self):
        return np.var(self.X, axis = 0)

    def get_median(self):
        return np.median(self.X, axis = 0)

    def get_min(self):
        return np.min(self.X, axis = 0)

    def get_max(self):
        return np.max(self.X, axis = 0)

    def summary(self):
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'variance': self.get_variance(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max()
             }
        )

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # matriz
    y = np.array([1, 2])  # vetor
    features = ['A', 'B', 'C', 'D']
    label = 'y'
    dataset = Dataset(X = x, y=y, features=features, label=label)
    print('shape:', dataset.shape())
    print('has label:', dataset.has_label())
    print('classes:', dataset.get_classes())
    print('mean:', dataset.get_mean())
    print('variance:', dataset.get_variance())
    print('median:', dataset.get_median())
    print('minimo:', dataset.get_min())
    print('summary:', dataset.summary())

