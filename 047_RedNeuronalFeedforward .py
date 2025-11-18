from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Cargar y preparar el dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Aplanar imágenes de 28x28 a vectores de 784 y normalizar
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 28*28).astype('float32') / 255.0

# One-hot encoding de las etiquetas
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# 2. Definir las entradas del modelo funcional
input_a = Input(shape=(28*28,), name='input_a')
input_b = Input(shape=(28*28,), name='input_b')

# 3. Bloque compartido
shared_dense = Dense(128, activation='relu')

processed_a = shared_dense(input_a)
processed_b = shared_dense(input_b)

# 4. Concatenar las dos ramas
concatenated = concatenate([processed_a, processed_b], axis=-1)

# 5. Capa de salida
output = Dense(10, activation='softmax')(concatenated)

# 6. Crear y compilar el modelo
model = Model(inputs=[input_a, input_b], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Opcional: ver la arquitectura
model.summary()

# 7. Entrenar el modelo
# Usamos la MISMA entrada dos veces, solo para ilustrar el uso de múltiples inputs
history = model.fit(
    [X_train, X_train],  # input_a y input_b
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 8. Evaluar el modelo
test_loss, test_acc = model.evaluate([X_test, X_test], y_test)
print(f'Test accuracy: {test_acc:.4f}')
