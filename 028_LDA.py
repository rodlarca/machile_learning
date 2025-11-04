# Importar las librerías necesarias
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Cargar el conjunto de datos de iris
iris = load_iris()
X = iris.data
y = iris.target

# Aplicar LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

# Visualización de LDA
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', s=50)
plt.title("Reducción de Dimensionalidad usando LDA")
plt.show()