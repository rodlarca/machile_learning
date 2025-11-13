# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Generar un dataset sintético
X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=0)

# Aplicar K-medias con 5 clusters
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Calcular la Inercia
inertia = kmeans.inertia_
print(f"Inercia: {inertia}")

# Calcular el Silhouette Score
sil_score = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {sil_score}")

# Calcular el Davies-Bouldin Index
db_index = davies_bouldin_score(X, y_kmeans)
print(f"Davies-Bouldin Index: {db_index}")

# Visualización de los clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red')
plt.title("Clusters Formados por K-Medias")
plt.show()