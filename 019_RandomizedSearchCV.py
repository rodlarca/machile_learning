import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='target')

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import loguniform

# Definir hiperparámetros a buscar
param_dist = [
    # L1 solo con liblinear o saga; SAGA suele requerir más iteraciones
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'],
     'C': loguniform(1e-3, 1e3), 'max_iter': [1000, 3000]},
    # L2 con liblinear (estable con pocas iteraciones)
    {'penalty': ['l2'], 'solver': ['liblinear'],
     'C': loguniform(1e-3, 1e3), 'max_iter': [200, 500]},
    # L2 con saga (más iteraciones)
    {'penalty': ['l2'], 'solver': ['saga'],
     'C': loguniform(1e-3, 1e3), 'max_iter': [1000, 3000]},
]

# Crear el modelo
model = LogisticRegression(tol=1e-3, random_state=42)

# Configurar RandomSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)

# Ejecutar RandomSearchCV
random_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

# Evaluar el mejor modelo en el conjunto de prueba
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del mejor modelo: {accuracy}')
print('Informe de Clasificación:')
print(classification_report(y_test, y_pred))

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(conf_matrix)