# -*- coding: utf-8 -*-

from si.data.dataset import Dataset
import numpy as np


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> tuple:
    '''
        Random splits a dataset into a train and a test set.

    :param dataset: Dataset object
    :param test_size: size of the test set.
    :param random_state: seed to feed the random permutations.

    :return tuple with training dataset and test dataset
    '''
    np.random.seed(random_state)

    # dá o tamanho do teste
    len_samples = dataset.shape()[0]
    len_test = int(test_size * len_samples)

    #dataset permutations
    permutations = np.random.permutation(len_samples)

    #dividir o teste e treino
    test_split = permutations[:len_test]
    train_split = permutations[len_test:]
    #passar para dataset
    train = Dataset(X=dataset.X[train_split], y=dataset.y[train_split], features=dataset.features, label=dataset.label)
    test = Dataset(X=dataset.X[test_split], y=dataset.y[test_split], features=dataset.features, label=dataset.label)

    return train, test