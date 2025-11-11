# Limpieza y Transformación de Datos

# Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar un conjunto de datos de ejemplo
data = pd.read_csv('dataset.csv')

# Limpiar datos: eliminar duplicados y manejar valores faltantes
data.drop_duplicates(inplace=True)
data.fillna(data.mean(), inplace=True)

# Normalizar características numéricas
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))

# Crear características polinómicas
poly = PolynomialFeatures(degree=2, include_bias=False)
data_poly = poly.fit_transform(data_scaled)

# Agregar las características polinómicas al DataFrame original
data_final = pd.DataFrame(data_poly, columns=poly.get_feature_names_out(data.columns))

print(data_final.head())

# Creación y Selección de Características

# Crear nuevas características utilizando operadores sobrecargados
data_final['Ratio_Feature1_Feature2'] = data_final['feature1'] / data_final['feature2']
data_final['Product_Feature3_Feature4'] = data_final['feature3'] * data_final['feature4']

# Seleccionar las características más importantes utilizando un modelo de bosque aleatorio
X = data_final.drop('target', axis=1)
y = data_final['target']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importances = model.feature_importances_
important_features = X.columns[importances > np.percentile(importances, 75)]

print(f"Características seleccionadas: {important_features}")

# Aplicación de Funciones Personalizadas

# Definir una función personalizada para escalar datos entre 0 y 1
def escalar_personalizado(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

# Aplicar la función personalizada a una característica específica
data_final['Feature_Scaled'] = data_final['feature1'].apply(escalar_personalizado, args=(data_final['feature1'].min(), data_final['feature1'].max()))

print(data_final['Feature_Scaled'].head())

# Entrenamiento y Evaluación del Modelo

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X[important_features], y, test_size=0.3, random_state=42)

# Entrenar un modelo de Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión del modelo: {accuracy:.4f}")