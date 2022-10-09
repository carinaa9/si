import numpy as np
import pandas as pd
import src.si.io.csv as csv
print('Exercício 1')
print()
# 1.1) CARREGAR FICHEIRO USANDO READ
print('1.1) Neste exercício, vamos usar o iris dataset. Carrega o iris.csv usando o método read apropriado para o tipo de ficheiro')
print()

filename = 'C:/Users/Asus/si/datasets/iris.csv'
df = csv.read_csv(filename, features=True, label=True)
print(df)
print()

# 1.2) SELECIONAR A PRIMEIRA VARIÁVEL INDEPENDENTE E VERIFICAR A DIMENSÃO DO ARRAY RESULTANTE
print('1.2) Seleciona a primeira variável independente e verifica a dimensão do array resultante')
print()
var_ind = df.X[:, 0] # primeira coluna que é a sepal_length

print('Dimension:', var_ind.shape)
print()

# 1.3) SELECIONAR AS ULTIMAS 5 AMOSTRAS DO IRIS, MEDIA PARA CADA VARIAVEL INDEPENDENTE/FEATURE
print('1.3) Seleciona as últimas 5 amostras do iris dataset. Qual a média das últimas 5 amostras para cada variável independente/feature?')
print()
#selecao
samples = df.X[-5:, :] #seleciona as ultimas 5 linhas e todas as colunas
print('Samples:', samples)
#media
mean_samples = np.nanmean(samples, axis=0)
print('Mean samples:', mean_samples)

# 1.4) SELCIONAR TODAS AS AMOSTRAS SUP OU = A 1 PARA TODAS AS FEATURES
print()
print('1.4) Seleciona todas as amostras do dataset com valor superior ou igual a 1. Nota que o array resultante deve ter apenas amostras com valores iguais ou superiores a 1 para todas as features.')
print()


sup = np.all(df.X > 1, axis=1) # seleciona todos dos dados do datasetX >= 1
sup = df.X[sup, :] # separar para cada features apenas
print('Dimension:', sup.shape)
print()

# SELECIONAR TODAS IRIS SETOSA

print('1.5) Seleciona todas as amostras com a classe/label igual a ‘Iris-setosa’. Quantas amostras obténs?')
print()
class_setosa = (df.y == 'Iris-setosa')
class_setosa = df.X[class_setosa, :]
print('Dimension:', class_setosa.shape)