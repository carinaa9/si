# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable, Dict
import numpy as np
from src.si.data.dataset import Dataset
from src.si.model_selection.split import train_test_split

def cross_validate(model, dataset: Dataset,
                   scoring: Callable = None,
                   cv: int = 3,
                   test_size: float = 0.2) -> Dict [str, list[float]]:

    # dicionario -- output esperado
    scores = {
        'seeds': [],
        'train': [],
        'test': []
    }

    for i in range (cv): # para cada cross validation
        random_state = np.random.randint(0, 1000)

        scores['seeds'].append(random_state)

        #dividir
        train, test = train_test_split(dataset= dataset, test_size = test_size, random_state=random_state)

        # fit do modelo no treino
        model.fit(train)

        #score do modelo no teste

        if scoring is None: # usa a accuracy que está implementada no nosso modelo
            #guardar score do treino
            scores['train'].append(model.score(train))

            # guardar score do teste
            scores['test'].append(model.score(test))

        else:
            # caso o utilizador dê uma funcção diferente de score (no nosso caso é a accuracy)
            y_train = train.y
            y_test = test.y

            #guardar de novo
            scores['train'].append(scoring(y_train, model.predict(train)))

            scores['test'].append(scoring(y_test, model.predict(test)))

    return scores

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.neighbors.knn_classifier import KNNClassifier

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the KNN
    knn = KNNClassifier(k=3)

    # cross validate the model
    scores_ = cross_validate(knn, dataset_, cv=5)

    # print the scores
    print(scores_)