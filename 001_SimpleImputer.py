from sklearn.impute import SimpleImputer
import pandas as pd

# Datos con valores faltantes
X = pd.DataFrame({
    'A': [1, 3, 7],
    'B': [2, None, 8],
    'C': [None, 6, 9]
})

print("Datos originales:")
print(X)

# Crear un imputador con la estrategia de la media
imputer = SimpleImputer(strategy='mean')

# Ajustar y transformar los datos
X_imputed = imputer.fit_transform(X)

# Convertir el resultado a un DataFrame de pandas
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

print("\nDatos imputados:")
print(X_imputed_df)