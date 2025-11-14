# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Generar un dataset sintético
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

# Crear el modelo de agrupamiento jerárquico aglomerativo
model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)

# Ajustar el modelo
model.fit(X)

# Generar el dendrograma
Z = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrograma de Agrupamiento Jerárquico")
plt.show()

# Generar el dendrograma con enlace completo
Z_complete = linkage(X, method='complete')
plt.figure(figsize=(10, 7))
dendrogram(Z_complete)
plt.title("Dendrograma con Enlace Completo")
plt.show()