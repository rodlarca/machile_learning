# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.datasets import make_blobs

# Crear un conjunto de datos sintético con señales mezcladas
X, _ = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
X = np.dot(X, np.random.RandomState(0).randn(2, 2))

# Aplicar ICA para separar las señales mezcladas
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X)

# Visualización de las señales originales y separadas
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue')
plt.title("Señales Mezcladas")

plt.subplot(1, 2, 2)
plt.scatter(X_ica[:, 0], X_ica[:, 1], c='red')
plt.title("Señales Separadas por ICA")
plt.show()