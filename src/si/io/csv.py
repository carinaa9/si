# -*- coding: utf-8 -*-

import pandas as pd
from src.si.data.dataset import Dataset
from typing import Optional, Union


def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    '''
    Reads csv file and returns a Dataset object of that file

    :param filename: name/path of file
    :param sep: separator between values. Defaults to ,
    :param features: If the csv file has feature names. Defaults to True
    :param label: If the dataset has defined labels receives an integer value that tells the column of the labels. Defaults to None
    '''

    data = pd.read_csv(filename, sep=sep)

    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None

    else:
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(X, y, features=features, label=label)


def write_csv(dataset, filename: str, sep: str = ',', features: Optional[bool] = True, label: Optional[bool] = True):
    '''
    Writes a csv file from a dataset object

    :param dataset: Dataset to save on csv format
    :param filename: Name of the csv file that will be saved
    :param sep: Separator of values. Defaults to ',''
    :param features: Boolean value that tells if the dataset object has feature names. Defaults to True.
    :param label: Boolean value that tells if the dataset object has label names Defaults to True
    '''

    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index = False)



if __name__ == '__main__':
    # file = "C:/Users/Asus/si/datasets/iris_missing_data.csv"
    # a = read_csv(filename=file, sep = ",", features=True, label=4)
    # print(a.print_dataframe())
    # print(a.summary())
    # write_csv(a, "csv_write.csv", features=True, label=False)

    file = 'C:/Users/Asus/si/datasets/iris_missing_data.csv'
    a = read_csv(filename=file, sep=",", features=True, label=4)
    print(a)

