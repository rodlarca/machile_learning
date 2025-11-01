import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

print(data.target_names)  # ['malignant' 'benign']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt

# Entrenar KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predicciones y probabilidades
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]  # probabilidad de clase positiva (maligno)

# Evaluar el modelo
print(f"Exactitud: {accuracy_score(y_test, y_pred):.2f}")
print("\nInforme de Clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

# CURVA ROC y AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_value = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Breast Cancer (KNN)')
plt.legend(loc='lower right')
plt.show()
