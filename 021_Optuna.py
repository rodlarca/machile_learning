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