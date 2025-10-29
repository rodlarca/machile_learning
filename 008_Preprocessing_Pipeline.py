import pandas as pd
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cargar el dataset Wine Quality
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Añadir valores faltantes para la práctica
import numpy as np
X.loc[0:10, 'alcohol'] = np.nan

# Definir transformaciones
numeric_features = X.select_dtypes(include=['float64', 'int']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# En este dataset no hay categóricas, pero se puede añadir una columna categórica ficticia para la práctica
X['quality'] = np.where(y > 1, 'high', 'low')
categorical_features = ['quality']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Integrar en un pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Aplicar preprocesamiento
X_transformed = pipeline.fit_transform(X)

print("Preprocesamiento completado. Datos transformados listos para modelar.")