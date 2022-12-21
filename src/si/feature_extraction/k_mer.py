# -*- coding: utf-8 -*-


import itertools

import numpy as np

from src.si.data.dataset import Dataset




class KMer:
    #conjunto de substrings de comprimento k contidas numa sequencia
    # especifico para o DNA (alfabeto: ACTG)

    '''
    A sequence descriptor that returns the k-mer composition of the sequence

    :param k: int. The k-mer length
   '''

    def __init__(self, k: int = 3, alphabet: str = 'DNA'):
        '''
        :param k: The k-mer length.
        '''
        # parameters . tamanho da substring
        self.k = k

        #alfabeto para DNA - da sequencia biologica - composição peptidica
        # pode ser dna (ACTG) ou aminoacido (_ACDEFGHIKLMNPQRSTVWY ou FLIMVSPTAY_HQNKDECWRG)
        self.alphabet = alphabet.upper() # caso o input seja minusculas

        if self.alphabet == 'DNA':
            self.alphabet = 'ACTG'
            #print('It is DNA')
        elif self.alphabet == 'PROTEIN':
            self.alphabet = '_ACDEFGHIKLMNPQRSTVWY'
            #print('It is a protein/peptide')
        else:
            self.alphabet = self.alphabet
            #print('That is not a protein/peptide or DNA')

        # attributes - parametros estimados
        # todos os kmers possiveis
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        '''
        Fits the descriptor to the dataset

        :param dataset: The dataset to fit the descriptor to
        :return KMer: The fitted descriptor
        '''
        # generate the k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        '''
        Calculates the k-mer composition of the sequence

        :param sequence : The sequence to calculate the k-mer composition for
        :return: The k-mer composition of the sequence
        '''

        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        # normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        '''
        Transforms the dataset

        :param dataset: The dataset to transform
        :return: The transformed dataset
        '''

        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence)
                                       for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        '''
        Fits the descriptor to the dataset and transforms the dataset

        :param dataset : The dataset to fit the descriptor to and transform
        :return: The transformed dataset
        '''
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')

    k_mer_ = KMer(k=3)
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features)