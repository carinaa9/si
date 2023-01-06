# -*- coding: utf-8 -*-

from typing import Tuple, Union, Callable
import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy


class StackingClassifier:
    # usa conjunto de modelos para gerar previsoes
    # previsoes usadas para treinar outro modelo (modelo final)
    #modelo final usado para prever a variavel de saida (Y)

    def __init__(self, models: list, final_model):
        '''
        :param models:  models
        :param final_model: the final model
        '''
        # conjunto de modelos
        self.models = models 

        #modelo final
        self.final_model = final_model 

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        '''
        Fit the models of ensemble

        :param dataset: Dataset to fit the models 

        :return: self: StackingClassifier
        '''

        # treino dos models (como o voting)
        for model in self.models:
            model.fit(dataset)

        # predictions modelo treinado
        # guardar as previsões de cada modelo
        predictions = np.array([model.predict(dataset) for model in self.models])
        
        # treino modelo final
        #passar para dataset as predictions anteriores em forma de array e transposta
        prev_model = Dataset(dataset.X, predictions.T)

        #treino do modelo final com as predictions anteriores
        final = self.final_model.fit(prev_model)
        return final

    def predict(self, dataset: Dataset) -> np.array:
        '''
        Estimates the output variable using the trained models and the final model
        
        :param dataset: Dataset to predict the labels
        '''
        # estimativa da variavel de saida usando os modelos treinados e o final
        #previsoes de cada modelo no conjunto de modelos

        # guarda as predictions para cada modelo
        predictions = np.array([model.predict(dataset) for model in self.models]) 
        #nao transpor pq vai ser no predict quando passar para dataset

        pred_final = Dataset(dataset.X, predictions.T)

        # treino do final com as predictions anteriores
        final_pred = self.final_model.predict(pred_final)
        return final_pred

    def score(self, dataset: Dataset) -> float:
        '''
        Calculates the accuracy of the model

        :param dataset: dataset to calculate the score
        :return: accuracy of the model
        '''
        # calculo do erro entre as previsoes e os valores reais
        
        return accuracy(dataset.y, self.predict(dataset))
