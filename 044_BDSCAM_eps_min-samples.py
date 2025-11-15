# Importar el conjunto de datos de iris
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

iris = load_iris()
X_iris = iris.data

# Aplicar DBSCAN
dbscan_iris = DBSCAN(eps=0.5, min_samples=5)
labels_iris = dbscan_iris.fit_predict(X_iris)

# Visualizar los resultados en dos dimensiones
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=labels_iris, cmap='viridis', s=50)
plt.title("DBSCAN en el Conjunto de Datos de Iris")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()