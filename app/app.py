import pandas as pd
from imblearn.over_sampling import SMOTE
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df_credit = pd.read_csv("C:/Users/DESENVOLVIMENTO/Documents/app/creditcard.csv", delimiter=(','))

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

#Treinamento o Modelo
model_rl_clf = model_rl_clf.fit(x_treino_normalizado, y_treino)

#
y_rl_pred = model_rl_clf.predict(x_teste_normalizado)

# Importe as bibliotecas Flask e jsonify
#from flask import Flask, request, jsonify, render_template

# Crie uma instância do aplicativo Flask
app = Flask(__name__, template_folder='C:/Users/DESENVOLVIMENTO/Documents/app/templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Rota para a página inicial
@app.route('/')
def index():
    return render_template('index.html')
    
# Rota para a ação de detecção de fraude
@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    # Obtenha os dados do formulário HTML
    # Certifique-se de que o nome dos campos no formulário corresponda aos nomes das colunas em seus dados
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])
    feature5 = float(request.form['feature5'])
    feature6 = float(request.form['feature6'])
    feature7 = float(request.form['feature7'])
    feature8 = float(request.form['feature8'])
    feature9 = float(request.form['feature9'])
    feature10 = float(request.form['feature10'])
    feature11 = float(request.form['feature11'])
    feature12 = float(request.form['feature12'])
    feature13 = float(request.form['feature13'])
    feature14 = float(request.form['feature14'])
    feature15 = float(request.form['feature15'])
    feature16 = float(request.form['feature16'])
    feature17 = float(request.form['feature17'])
    feature18 = float(request.form['feature18'])
    feature19 = float(request.form['feature19'])
    feature20 = float(request.form['feature20'])
    feature21 = float(request.form['feature21'])
    feature22 = float(request.form['feature22'])
    feature23 = float(request.form['feature23'])
    feature24 = float(request.form['feature24'])
    feature25 = float(request.form['feature25'])
    feature26 = float(request.form['feature26'])
    feature27 = float(request.form['feature27'])
    feature28 = float(request.form['feature28'])
    feature29 = float(request.form['feature29'])
    feature30 = float(request.form['feature30'])

    # Realize a previsão com o modelo
    prediction = model_rl_clf.predict([[feature1, feature2,feature3,feature4,feature5,feature6,feature7,feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,feature25, feature26, feature27, feature28, feature29, feature30]])[0]

    # Determine o resultado (fraude ou não fraude)
    if prediction == 1:
        result = "Fraude"
    else:
        result = "Não Fraude"

    # Retorne o resultado como uma resposta JSON
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)