# -*- coding: utf-8 -*-

import pandas as pd

def read_csv(filename, sep, features = None, label = None):
    '''Lê o ficheiro especificado e retorna um Dataset'''
    pd.read_csv(filename, sep, features, label)
    pass






def write_csv(filename,dataset, sep, features = None, label = None):
    '''escreve o ficheiro especificado com os argumentos indicados'''
    file = pd.write_csv(filename, dataset, sep, features, label)

    pass



if __name__ == '__main__':
    print(':', dataset.read_csv())
