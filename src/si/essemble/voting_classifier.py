# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable
import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy


class VotingClassifier:
    '''
    Ensemble classifier that uses the majority vote to predict the class labels.

    :param models : array-like, shape = [n_models]
        Different models for the ensemble
    '''
    def __init__(self, models):
        """
        Initialize the ensemble classifier

        :param models: Different models for the ensemble
        """
        # parameters
        # conjunto de modelos
        self.models = models

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        '''
        Fit the models according to the given training data.

        :param dataset : The training data
        :return self : The fitted model
        '''
        # treinar os modelos do ensemble
        for model in self.models:
            model.fit(dataset)

        #retorna ele proprio
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        Predict class labels for samples in X.

        :param dataset : The test data
        :return y : The predicted class labels
        '''
        # predict: estima a variavel de saida usando os modelos treinados e a função de combinação
        # previsoes de cada modelo
        #combina as previsoes de cada modelo usando a técnica de votação
        # classe mais representada é escolhida



        # helper function
        def _get_majority_vote(pred: np.ndarray) -> int:
            '''
            It returns the majority vote of the given predictions

            :param pred: The predictions to get the majority vote of
            :return majority_vote: The majority vote of the given predictions
            '''
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose() 
        return np.apply_along_axis(_get_majority_vote, axis = 1, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        '''
        Returns the mean accuracy on the given test data and labels.

        :param dataset: The test data
        :return score: Mean accuracy
        '''
        # calculo do erro entre as previsoes e os valores reais
        # usa a accuracy como medida de erro
        # estima valores de saida(Y) usando modelos treinados (predict)

        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.neighbors.knn_classifier import KNNClassifier
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f'Score: {score}')

    print(voting.predict(dataset_test))