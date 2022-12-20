# -*- coding: utf-8 -*-
import itertools
from typing import Tuple, Union, Callable, Dict
import numpy as np
from src.si.data.dataset import Dataset
from src.si.model_selection.cross_validate import cross_validate

def grid_search(model,
                dataset: Dataset,
                parameter_grid: Dict[str, Tuple],
                scoring: Callable = None,
                cv: int = 3,
                test_size: float = 0.2) -> Dict[str, list[float]]:

    for parameter in parameter_grid:
        # verificar se o modelo do utilizador retorna um parametro atraves de true or false
        if not hasattr(model, parameter):
            raise AttributeError(f'Model {model} does not have parameter {parameter}.')

    scores = []

    for combination in itertools.product(*parameter_grid.values()): # asterisco faz unpacking e não altera a ordem
        #retorna a lista de combinações

        #para guardar os parametros
        parameters = {} # exemplo: parameter: {'l2_penalty'= 2, 'alpha'= 0,01, etc}

        for parameter, value in zip(parameter_grid.keys(), combination):
            # faz iteração de duas 'sequencias' ao mesmo tempo e atribui ao parameter o primeiro e ao value o segundo
            setattr(model, parameter, value)
            parameters[parameter] = value

        #validacao cruzado do modelo
        score = cross_validate(model = model, dataset= dataset, scoring = scoring, cv = cv, test_size = test_size)

        #add the parameter configuration
        score['parameters'] = parameters

        #guardar o score
        scores.append(score)

    return scores




if __name__ == '__main__':
    # import dataset
    from src.si.data.dataset import Dataset
    from src.si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores_ = grid_search(knn,
                             dataset_,
                             parameter_grid=parameter_grid_,
                             cv=3)

    # print the scores
    print(scores_)



