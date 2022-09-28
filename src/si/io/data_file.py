# -*- coding: utf-8 -*-

import numpy as np

def read_data_file(filename, sep, label = None):
    '''Lê o ficheiro especificado e retorna um Dataset'''
    np.genfromtxt(filename, sep, label)

    pass


def write_data_file(filename, dataset, sep, label = None):
    '''escreve o ficheiro especificado com os argumentos indicados'''
    np.savetxt(filename, dataset, sep, label)

    pass