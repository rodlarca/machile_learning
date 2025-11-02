import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset de Breast Cancer
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='target')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def objective(trial):
    # Definir los hiperparámetros a optimizar
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    #C = trial.suggest_float('C', 1e-4, 1e2)
    C = trial.suggest_float('C', 1e-3, 1e1, log=True)
    #solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    solver = trial.suggest_categorical('solver', ['liblinear'])
    max_iter = trial.suggest_int('max_iter', 1000, 3000)
    
    # Crear el modelo de Regresión Logística con los hiperparámetros sugeridos
    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, random_state=42)
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular la exactitud
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Crear un estudio de Optuna y optimizar la función objetivo
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(study.best_params)

# Mostrar la mejor exactitud obtenida
print(f"Mejor exactitud obtenida: {study.best_value}")

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Obtener el mejor modelo
best_params = study.best_params
best_model = LogisticRegression(**best_params, random_state=42)

# Entrenar el mejor modelo
best_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del mejor modelo: {accuracy}')
print('Informe de Clasificación:')
print(classification_report(y_test, y_pred))

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(conf_matrix)

# Curva ROC y AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Plotear la curva ROC
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()