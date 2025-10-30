import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('datasets/Housing.csv')

# Separar variables predictoras y objetivo
X = data[['sqft_living', 'sqft_lot']]
y = data['price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalado de datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Validación cruzada para encontrar el mejor valor de K
k_values = list(range(1, 21))   # Probar K de 1 a 20
mse_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    # Validación cruzada con 5 particiones
    scores = cross_val_score(knn, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mse_scores.append(scores.mean())

# Seleccionar el K con el mayor valor (menos error negativo)
best_k = k_values[mse_scores.index(max(mse_scores))]
print(f'Mejor valor de K: {best_k}')

# Entrenar el modelo final con el mejor K
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Error Cuadrático Medio: {mse:.2f}')
print(f'R-cuadrado: {r2:.4f}')

# Visualizar cambio MSE para diferentes valores de K.
plt.plot(k_values, mse_scores)
plt.xlabel('Número de Vecinos (K)')
plt.ylabel('MSE Negativo')
plt.title('Selección del Mejor K')
plt.show()
