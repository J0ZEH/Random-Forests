# Importando as bibliotecas necessárias
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregando os dados de treinamento e teste
def carregar_dados():
    # Código para carregar os dados do conjunto de treinamento e teste
    # Certifique-se de ter uma matriz 'X' contendo os recursos e um vetor 'y' contendo os rótulos de destino
    X = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12],
         [13, 14, 15],
         [16, 17, 18]]

    y = [0, 1, 0, 1, 0, 1]

    return X, y

# Dividindo os dados em conjunto de treinamento e teste
def dividir_dados(X, y):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_treino, X_teste, y_treino, y_teste

# Treinando o modelo
def treinar_modelo(X_treino, y_treino):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)
    return modelo

# Avaliando o modelo
def avaliar_modelo(modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    return acuracia

# Executando o código principal
def principal():
    # Carregando os dados
    X, y = carregar_dados()

    # Dividindo os dados em conjunto de treinamento e teste
    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y)

    # Treinando o modelo
    modelo = treinar_modelo(X_treino, y_treino)

    # Avaliando o modelo
    acuracia = avaliar_modelo(modelo, X_teste, y_teste)
    print(f"Acurácia do modelo: {acuracia}")

# Executando o código principal
if __name__ == "__main__":
    principal()





    