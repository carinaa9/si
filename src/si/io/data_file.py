# -*- coding: utf-8 -*-
import sys
import numpy as np
from src.si.data.dataset import Dataset
from typing import Union



def read_data_file(filename: str, sep: str = ",", label: Union[None, int] = None):
    '''
    Reads a data file and returns a dataset object

    :param filename: File name of path.
    :param sep: Separator between values. Defaults to ','.
    :param label: Where are the labels. Defaults to None.
    '''

    raw_data = np.genfromtxt(filename, delimiter=sep)

    if label:
        #separar o dataset
        X = raw_data[:, :-1] # todas as linhas, todas as colunas menos a última
        y = raw_data[:, -1] # todas as linhas , ultima coluna

    else:
        X = raw_data
        y = None

    return Dataset(X, y)


def write_data_file(dataset, filename: str, sep: str = ",", label: Union[None, int] = None):
    '''
    Writes a dataset into a cvs file

    :param dataset: The dataset that will be written.
    :param filename: The filename or path for the file that will be written
    :param sep: The separator between values. Defaults to ",".
    :param label: Where the defined labels are. Defaults to None.
    '''
    if label is not None:
        dataset = np.append(dataset.X, dataset.y[:, None], axis=1)
    else:
        dataset = dataset.X

    np.savetxt(fname=filename,X =dataset, delimiter=sep)


if __name__ == '__main__':
    file = 'C:/Users/Asus/si/datasets/iris_missing_data.csv'
    a = read_data_file(file, label=4)
    write_data_file(a, 'write_data_file.csv', label=4)