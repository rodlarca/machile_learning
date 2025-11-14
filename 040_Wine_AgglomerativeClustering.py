# Importar el conjunto de datos de vinos
from sklearn.datasets import load_wine
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

wine = load_wine()
X_wine = wine.data

# Aplicar el agrupamiento jerárquico con enlace de Ward
model_wine = AgglomerativeClustering(n_clusters=3)
labels_wine = model_wine.fit_predict(X_wine)

# Visualizar los resultados utilizando un par de características
plt.scatter(X_wine[:, 0], X_wine[:, 1], c=labels_wine, cmap='viridis')
plt.title("Agrupamiento Jerárquico en el Conjunto de Datos de Vinos")
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.show()