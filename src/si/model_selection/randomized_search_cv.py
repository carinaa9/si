# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable, Dict
import numpy as np
from src.si.data.dataset import Dataset
from src.si.model_selection.cross_validate import cross_validate


def randomized_search_cv(model, 
                        dataset:Dataset, 
                        parameter_distribution: Dict[str, float],
                        scoring: Callable = None, cv: int = 3, 
                        n_iter: int = 1000,
                        test_size: float = 0.2) -> Dict[str, list[float]]:
    '''
    It allows you to credibly evaluate the performance of a model on a given dataset

    :param model: model to cross validate
    :param dataset: a given dataset
    :param scoring: score
    :param cv: number of folds
    :param test_size: the size of the test set
    '''

    #lista de dicionarios com chaves de:
    # parameters --> combinação de parametros 
    # seeds --> seeds geradas para cada fold 
    # train --> scores do modelo no dataset de treino para cada fold 
    # test --> scores do modelo no dataset de teste para cada fold 
    scores = {
        'parameters': [],
        'seeds': [],
        'train': [],
        'test': []
    }
    

    #  checks if parameters exist in the model
    for parameter in parameter_distribution:
        # verificar se o modelo do utilizador retorna um parametro atraves de true or false
        if not hasattr(model, parameter):
            raise AttributeError(f'Model {model} does not have parameter {parameter}.')

    
    #  obter n_iter combinações de parametros
    for combination in range(n_iter):
        #retorna a lista de combinações
        
        # random seed
        random_state = np.random.randint(0, 1000)

        # guardar cada seed
        scores['seeds'].append(random_state)

        #para guardar os parametros
        parameters = {} # exemplo: parameter: {'l2_penalty'= 2, 'alpha'= 0,01, etc}

        # set the parameters
        for parameter, value in parameter_distribution.items():
           #retirar um valor aleatorio da distribuição de valores de cada parametro
            parameters[parameter] = np.random.choice(value)

        # set the parameters to the model
        for parameter, value in parameters.items():
            setattr(model, parameter, value)

        # cross validation para obter os scores
        score = cross_validate(model = model, dataset = dataset, scoring = scoring, cv = cv, test_size = test_size)

        # guardar no dict and return the scores 
        scores['parameters'].append(parameters)
        scores['train'].append(score['train'])
        scores['test'].append(score['test'])

    return scores
