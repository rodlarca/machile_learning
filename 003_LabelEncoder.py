from sklearn.preprocessing import LabelEncoder

# Datos categóricos
y = ['rojo', 'verde', 'azul', 'verde', 'rojo']

# Crear un codificador de etiquetas
encoder = LabelEncoder()

# Ajustar y transformar los datos
y_encoded = encoder.fit_transform(y)

print(y_encoded)