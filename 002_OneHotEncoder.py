from sklearn.preprocessing import OneHotEncoder

# Datos categóricos
X = [['rojo'], ['verde'], ['azul'], ['verde'], ['rojo']]

# Crear un codificador One-Hot
encoder = OneHotEncoder()

# Ajustar y transformar los datos
X_encoded = encoder.fit_transform(X)

print(X_encoded.toarray())