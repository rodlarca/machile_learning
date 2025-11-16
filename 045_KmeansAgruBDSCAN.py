# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Generar un conjunto de datos sintético
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

# Aplicar K-medias
kmeans = KMeans(n_clusters=4)
labels_kmeans = kmeans.fit_predict(X)

# Aplicar Agrupamiento Jerárquico
agglo = AgglomerativeClustering(n_clusters=4)
labels_agglo = agglo.fit_predict(X)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# Visualizar los resultados de K-medias
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50)
plt.title("K-medias")

# Visualizar los resultados de Agrupamiento Jerárquico
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_agglo, cmap='viridis', s=50)
plt.title("Agrupamiento Jerárquico")

# Visualizar los resultados de DBSCAN
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', s=50)
plt.title("DBSCAN")
plt.show()