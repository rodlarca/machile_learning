from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
X, y = fetch_openml('tic-tac-toe', version=1, return_X_y=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un imputador para manejar valores faltantes
imputer = SimpleImputer(strategy='most_frequent')

# Crear un codificador One-Hot para variables categóricas
encoder = OneHotEncoder()

# Preprocesar los datos de entrenamiento
X_train_imputed = imputer.fit_transform(X_train)
X_train_encoded = encoder.fit_transform(X_train_imputed)

# Preprocesar los datos de prueba
X_test_imputed = imputer.transform(X_test)
X_test_encoded = encoder.transform(X_test_imputed)

# Codificar las etiquetas de destino
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Crear y entrenar un clasificador de árboles de decisión
classifier = DecisionTreeClassifier()
classifier.fit(X_train_encoded, y_train_encoded)

# Hacer predicciones en el conjunto de prueba
y_pred = classifier.predict(X_test_encoded)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Precisión del modelo: {accuracy}")