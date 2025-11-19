from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

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
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Entrenamiento con callback
model.fit(X_train, y_train,
          epochs=50,               # se detendrá antes automáticamente
          batch_size=32,
          validation_split=0.2,
          callbacks=[early_stopping])

# Evaluación
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
