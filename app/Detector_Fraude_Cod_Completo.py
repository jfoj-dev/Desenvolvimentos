#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:35:28 2023

@author: Fernando Oliveira
"""
#%%Bibliotecas Utilizadas
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import socket
import warnings
warnings.filterwarnings('ignore')
import pickle

#%%Bibliotecas utilizadas no Pre-Processamento dos Dados
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, plot_roc_curve

#%%Bibliotecas utilizadas na construcao de Maquinas Preditivas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
from socket import socket

#%%Bibliotecas utilizadas na Avaliacao das Maquinas
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


#%% Carregando os dados
df_credit = pd.read_csv('/Users/mac/Desktop/TCC USP/Base_TCC/creditcard.csv', delimiter=(','))

df_credit

#%% Visualizando as primeiras 5 linhas do dataframe
df_credit.head()

#%% Informações Gerais da Fonte de Dados
df_credit.info()

#%% Verificando a existencia de variaveis missing
df_credit.isna().sum()

#%% Contando os valores da variavel categorica
df_credit['Class'].value_counts()


#%% Countplot Fraude ou Não Fraude
plt.title("Fraude ou Não Fraude")
sns.countplot(df_credit['Class'])
plt.show

#%% Estatísticas descritivas das variáveis
df_credit.describe()

#%%
df_nao_fraude = df_credit.Amount[df_credit.Class == 0]

df_nao_fraude.describe()

#%%
fraude = df_credit.Amount[df_credit.Class == 1]

fraude.describe()



#%% Separando os recursos (features) e o alvo (target)
x = df_credit.drop(['Class'], axis=1)
y = df_credit['Class']

#%%
corrmat = df_credit.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(df_credit[top_corr_features].corr(), annot=True, cmap="RdYlGn")

#%% 
features = list(df_credit.drop(columns=['Class']).columns) 

#%% Criando o balanceador SMOTE
smtb = SMOTE()

#%% Aplicando o balanceador
x,y = smtb.fit_resample(x, y)

np.bincount(y)

#%% Visualizando o balanceamento da variavel Class
ax= sns.countplot(x=y)

#%%Agrupando os valores para visualizar a quantidade de dados como True e False
df_credit.groupby('Class').size()

#%% Dividindo os dados em treinamento e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=42)

#%% Quantidade de observacoes em cada amostra
print("Shape do dataset de treino: ", x_treino.shape)
print("Shape do dataset teste: ", x_teste.shape)
#%% Normalizando os recursos
Normalizador = MinMaxScaler(feature_range=(0,1))
x_treino_normalizado = Normalizador.fit_transform(x_treino)
x_teste_normalizado = Normalizador.transform(x_teste)

#%% Quantidade de observações em cada amostra
print("Shape do dataset de treino: ", x_treino_normalizado.shape)
print("Shape do dataset de teste: ", x_teste_normalizado.shape)

#%% Criando o modelo de Regressao Logistica
model_rl_clf = LogisticRegression(random_state=42)

#%% Treinamento do Modelo 
model_rl_clf = model_rl_clf.fit(x_treino_normalizado, y_treino)

#%% Predicao da Regressao Logistica
y_rl_pred = model_rl_clf.predict(x_teste_normalizado)


#%% Gerando a matriz de confusão

## Compara os valores preditos com os valores observados (teste)

cm = confusion_matrix(y_teste, 
                      y_rl_pred, 
                      labels=model_rl_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model_rl_clf.classes_)

disp.plot()

#%% Avaliando o modelo
print('Classification metrics: \n', classification_report(y_teste, y_rl_pred))
print('Acuracia: \n', accuracy_score(y_teste, y_rl_pred))
#print('Confusion Matrix: \n', confusion_matrix(y_teste, mod_rl_pred))

#%% Plotar a curva ROC nos dados de teste

fpr, tpr, thresholds = roc_curve(y_teste,y_rl_pred)
roc_auc = auc(fpr, tpr)

gini = (roc_auc - 0.5)/(0.5)

plot_roc_curve(model_rl_clf, x_teste_normalizado, y_teste) 
plt.title('Coeficiente de GINI: %g' % round(gini,4), fontsize=12)
plt.xlabel('1 - Especificidade', fontsize=12)
plt.ylabel('Sensitividade', fontsize=12)
plt.show()





#%% Criando o modelo de Arvore de Decisão
model_tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)

#%% Treinamento do Modelo
model_tree_clf = model_tree_clf.fit(x_treino_normalizado, y_treino)

#%% Plotando a arvore (base de treino)
fig = plt.figure(figsize=(60,40))

_ = tree.plot_tree(model_tree_clf, 
                   feature_names=features,  
                   class_names=['Nao Fraude','Fraude'],
                   filled=True)

#%% Predicao com Arvore de Decisao
model_tree_pred = model_tree_clf.predict(x_teste_normalizado)

#%% Verificando a importância de cada variável do modelo

importancia_features = pd.DataFrame({'features':features,
                                     'importance':model_tree_clf.feature_importances_})

print(importancia_features)

#%% Gerando a matriz de confusão

## Compara os valores preditos pela árvore com os valores observados (teste)

cm = confusion_matrix(y_teste, 
                      model_tree_pred, 
                      labels=model_tree_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model_tree_clf.classes_)

disp.plot()


#%% validando o modelo
print('Classification metrics: \n', classification_report(y_teste, model_tree_pred))
print('Acuracia: \n', accuracy_score(y_teste, model_tree_pred))
#print('Confusion Matrix: \n', confusion_matrix(y_teste, tree_pred))

#%% Plotar a curva ROC nos dados de teste

fpr, tpr, thresholds = roc_curve(y_teste, model_tree_pred)
roc_auc = auc(fpr, tpr)

gini = (roc_auc - 0.5)/(0.5)

plot_roc_curve(model_tree_clf, x_teste_normalizado, y_teste) 
plt.title('Coeficiente de GINI: %g' % round(gini,4), fontsize=12)
plt.xlabel('1 - Especificidade', fontsize=12)
plt.ylabel('Sensitividade', fontsize=12)
plt.show()




#%% Alterando para entropia

model_tree_clf_entropia = DecisionTreeClassifier(max_depth=4, random_state=42, criterion="entropy")
model_tree_clf_entropia.fit(x_treino_normalizado, y_treino)

#%% Plotando a árvore

fig = plt.figure(figsize=(40,40))

_ = tree.plot_tree(model_tree_clf_entropia, 
                   feature_names=features,  
                   class_names=['Não Fraude','Fraude'],
                   filled=True)

#%% Previsoes com Arvore de Decisao
tree_pred_entropia = model_tree_clf_entropia.predict(x_teste_normalizado)

#%% Verificando a importância de cada variável do modelo

importancia_featuresace = pd.DataFrame({'features':features,
                                     'importance':model_tree_clf_entropia.feature_importances_})

print(importancia_features)

#%% Gerando a matriz de confusão

## Compara os valores preditos pela árvore com os valores observados (teste)

cm = confusion_matrix(y_teste, 
                      tree_pred_entropia, 
                      labels=model_tree_clf_entropia.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model_tree_clf_entropia.classes_)

disp.plot()


#%%
print('Classification metrics: \n', classification_report(y_teste, tree_pred_entropia))
print('Acuracia: \n', accuracy_score(y_teste, tree_pred_entropia))
#print('Confusion Matrix: \n', confusion_matrix(y_teste, tree_pred))

#%% Plotar a curva ROC nos dados de teste

fpr, tpr, thresholds = roc_curve(y_teste,tree_pred_entropia)
roc_auc = auc(fpr, tpr)

entropy = (roc_auc - 0.5)/(0.5)

plot_roc_curve(model_tree_clf_entropia, x_teste_normalizado, y_teste) 
plt.title('Coeficiente de Entropia: %g' % round(entropy,4), fontsize=12)
plt.xlabel('1 - Especificidade', fontsize=12)
plt.ylabel('Sensitividade', fontsize=12)
plt.show()