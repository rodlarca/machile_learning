from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Cargar y preprocesar MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28*28).astype('float32') / 255
X_test  = X_test.reshape(-1, 28*28).astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Modelo con Dropout
model = Sequential([
    Input(shape=(28*28,)),
    Dense(128, activation='relu'),
    Dropout(0.5),          # Apaga el 50% de neuronas durante el entrenamiento
    Dense(64, activation='relu'),
    Dropout(0.5),          # Reduce sobreajuste
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
model.fit(X_train, y_train,
          epochs=10,
          batch_size=32,
          validation_split=0.2)

# Evaluación
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
