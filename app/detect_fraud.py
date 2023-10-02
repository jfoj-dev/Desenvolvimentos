import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df_credit = pd.read_csv("C:/Users/DESENVOLVIMENTO/Desktop/App Web/creditcard.csv", delimiter=(','))

x= df_credit.drop(['Class'], axis=1)
y= df_credit['Class']

#Criando o balanceador SMOTE
smtb = SMOTE()

# Aplicando o balanceador
x,y = smtb.fit_resample(x, y)

# Dividindo os dados em treinamento e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=42)

# Normalizando os recursos
Normalizador = MinMaxScaler(feature_range=(0,1))
x_treino_normalizado = Normalizador.fit_transform(x_treino)
x_teste_normalizado = Normalizador.transform(x_teste)

# Criando o modelo de Regressao Logistica
model_rl_clf = LogisticRegression(random_state=42)

#Treinamento do Modelo 
model_rl_clf = model_rl_clf.fit(x_treino_normalizado, y_treino)
