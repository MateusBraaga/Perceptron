# Exercício - Decisão de ir ao Parque 
# Importando biblioteca
from sklearn.linear_model import Perceptron

# Dados de entrada (Ensolarado, Final de Semana, Parque Lotado)
X = [
    [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], 
    [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
]

# Saídas desejadas (Ir ao Parque?)
Y = [0, 1, 1, 1, 0, 0, 0, 0]

# Criando e treinando o perceptron
modelo = Perceptron()
modelo.fit(X, Y)

# Testando o modelo
print("Previsões para Ir ao Parque:")
testes = X
for teste in testes:
    previsao = modelo.predict([teste])
    print(f"Ensolarado: {teste[0]}, Final de Semana: {teste[1]}, Parque Lotado: {teste[2]} => Ir ao Parque? {'Sim' if previsao[0] == 1 else 'Não'}")



# Exercício 3 - Decisão sobre comer fora ou cozinhar em casa 
# Importando biblioteca
from sklearn.linear_model import Perceptron

# Dados de entrada (Cansado, Ingredientes em casa, Restaurante aberto, Pagamento recente)
X = [
    [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [0, 0, 1, 0], 
    [1, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]
]

# Saídas desejadas (Comer fora?)
Y = [0, 1, 0, 1, 1, 0, 0, 0]

# Criando e treinando o perceptron
modelo = Perceptron()
modelo.fit(X, Y)

# Testando o modelo
print("\nPrevisões para Comer Fora:")
testes = X
for teste in testes:
    previsao = modelo.predict([teste])
    print(f"Cansado: {teste[0]}, Ingredientes em casa: {teste[1]}, Restaurante aberto: {teste[2]}, Pagamento recente: {teste[3]} => Comer Fora? {'Sim' if previsao[0] == 1 else 'Não'}")
