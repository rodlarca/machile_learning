# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generar un conjunto de datos sintético con forma de media luna
X, _ = make_moons(n_samples=300, noise=0.05)

# Aplicar DBSCAN con parámetros iniciales
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title("Resultados de DBSCAN con eps=0.2 y min_samples=5")
plt.show()

# Aplicar DBSCAN con un valor de eps mayor
dbscan_eps_high = DBSCAN(eps=0.3, min_samples=5)
labels_eps_high = dbscan_eps_high.fit_predict(X)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels_eps_high, cmap='viridis', s=50)
plt.title("Resultados de DBSCAN con eps=0.3 y min_samples=5")
plt.show()

# Aplicar DBSCAN con un valor de min_samples mayor
dbscan_min_samples_high = DBSCAN(eps=0.2, min_samples=10)
labels_min_samples_high = dbscan_min_samples_high.fit_predict(X)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels_min_samples_high, cmap='viridis', s=50)
plt.title("Resultados de DBSCAN con eps=0.2 y min_samples=10")
plt.show()