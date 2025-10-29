import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(0)
edad = np.random.randint(20, 70, 100)
ingreso_anual = np.random.randint(30000, 100000, 100)
gasto_anual = 5000 + 0.3 * edad + 0.5 * ingreso_anual + np.random.randn(100) * 10000

# Crear DataFrame
df = pd.DataFrame({'Edad': edad, 'Ingreso Anual': ingreso_anual, 'Gasto Anual': gasto_anual})

# Separar variables predictoras y objetivo
X = df[['Edad', 'Ingreso Anual']]
y = df['Gasto Anual']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error Cuadrático Medio: {mse}')
print(f'R-cuadrado: {r2}')

# Gráfico de dispersión de las predicciones vs. valores reales
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs. Valores Reales')
plt.show()