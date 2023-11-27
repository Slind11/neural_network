import numpy as np 

#Функция активации сигмоида
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Входные данные
inputs = np.array([0.5, 0.3, 0.2])

#Веса
weights = np.array([-0.4, 0.3, -0.1])

# Сумматор (взвешенная сумма входов)
weighted_sum = np.dot(inputs, weights)

# Применение функции активации
output = sigmoid(weighted_sum)

print("Выходное значение нейрона:", output)

