import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Escalar las características
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_train = scaler.fit_transform(X_train)  # fit en train
#X_test = scaler.fit_transform(X_test) 
#X_test  = scaler.transform(X_test)       # solo transform en test

# Crear y entrenar el modelo
model = xgb.XGBClassifier(eval_metric="mlogloss", random_state=42)
model.fit(X_train, y_train) 

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud: {accuracy}')
print('Informe de Clasificación:')
print(classification_report(y_test, y_pred)) 

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(conf_matrix)