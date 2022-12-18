# -*- coding: utf-8 -*-
from typing import Tuple, Union
from typing import Callable
import pandas as pd
import numpy as np
from src.si.data.dataset import Dataset
from src.si.statisics.f_classification import f_classification


class SelectPercentile:

    def __init__(self, percentile: float = 0.25, score_func: Callable = f_classification) -> None:
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        '''
            Estimates the F and p for each feature using the scoring_func

        :param dataset: a given dataset
        :return: self
        '''
        self.F, self.p = self.score_func(dataset)

        return self

    def transform(self, dataset: Dataset) -> Dataset:
        '''
           Selects the features with the highest F value up to the indicated percentile.
           (for a dataset with 10 features and a 50% percentile, the transform should select
           the 5 features with higher F value)

        :param dataset: a given dataset
        :return: dataset
        '''
        length = len(dataset.features)
        percentile_mask = int(length * self.percentile)
        idxs = np.argsort(self.F)[-percentile_mask:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        '''
            Runs the fit and then the transform

        :param dataset: a given dataset
        :return: transformed dataset
        '''
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":
    percentile = SelectPercentile(0.50)
    dataset = Dataset(X=np.array([[0, 1, 2, 3],
                                  [0, 2, 4, 6],
                                  [1, 3, 5, 7]]),
                      y=np.array([0, 1, 2]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    percentile = percentile.fit_transform(dataset)
    print(dataset.features)
    print(percentile.features)