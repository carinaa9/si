# -*- coding: utf-8 -*-

from typing import Callable

import numpy as np



class SoftMaxActivation:

    '''
    This layer must calculate the probability of occurrence of each class
    using the formula

    Multiclass problems

    '''
    def __init__(self) -> None:
        pass

    def soft_max_calculate(self, x : np.ndarray) -> np.ndarray:
        '''
        Compute softmax values for each sets of scores in x

        :param x: a given data
        :return soft_max_calc: the probability of occurrence of each class
        '''
        
        #z_i ---> 𝑋−max(𝑋)
        # e_zi ---> exponencial do vetor z i
        # somatorio 𝑗𝐾_𝑒𝑧 ---> soma da exponencial do vetor z_i 
        # considera a seguinte função do numpy : np.sum (…, axis =1, keepdims True)
        
        # FORMULA e_zi/ somatorio e_zi sendo que e_zi é exponencial do z_i que é o maximo de x

        exp_x = np.exp(x - np.max(x))
        soft_max_calc = exp_x / np.sum(exp_x, axis = 1, keepdims = True)

        return soft_max_calc
