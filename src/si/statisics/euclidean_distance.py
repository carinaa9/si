# -*- coding: utf-8 -*-

import numpy as np
def euclidean_distance(x:np.ndarray, y:np.ndarray)-> np.ndarray:
    '''
    Calculates the euclidean distance between two variables

    :param x: numpy array- 1 dimension
    :param y: numpy array- 2+ dimensions
    :return distance: Distance between x and y samples
    '''
    # x = [1, 2, 3,..., n]
    # y = [[0, 1, 2, 3, ..., n]]
    euclidean_distance_value = np.sqrt(((x - y) ** 2).sum(axis=1)) # soma um a um, da as distancias entre os pontos x11 a y11, etc
    return euclidean_distance_value