from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Datos de ejemplo
X = [[1, 'rojo', 10],
     [2, 'azul', None],
     [None, 'verde', 12],
     [4, 'rojo', 8]]

# Definir las columnas numéricas y categóricas
numeric_features = [0, 2]
categorical_features = [1]

# Crear transformadores para cada subconjunto de columnas
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Crear un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Aplicar el preprocesamiento
X_transformed = preprocessor.fit_transform(X)

print(X_transformed)