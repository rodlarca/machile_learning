# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

# Cargar conjunto de datos
data = load_wine()
X = data.data
y = data.target

# Aplicar ACP
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualización de los resultados
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.title("Visualización de los datos de vino utilizando ACP")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()