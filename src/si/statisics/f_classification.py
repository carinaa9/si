# -*- coding: utf-8 -*-

from typing import Tuple, Union
import numpy as np
from scipy import stats
from si.data.dataset import Dataset

def f_classification(self, dataset:Dataset) -> Tuple:
    '''
        Groups the samples or examples by classes. Computes one-way ANOVA F-value for the provided dataset

    :param dataset: a given dataset
    :return: F and p values
    '''
    classes = dataset.get_classes()  # vou buscar as classes
    groups = [dataset.X[dataset.y == c] for c in classes] # para a 0 vou buscar todas as amostras no y que tem label 0, para 1, 2 e por ai fora
    F, p = stats.f_oneway(*groups) #uma sequencia de objetos/np arrays daí o asterisco

    return F, p