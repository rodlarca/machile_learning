# Importar las librerías necesarias
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

# Cargar el conjunto de datos de iris
iris = load_iris()
X = iris.data
y = iris.target

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Visualización de t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=50)
plt.title("Reducción de Dimensionalidad usando t-SNE")
plt.show()