from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Cargar dataset
max_features = 10000   # solo las 10.000 palabras más frecuentes
maxlen = 200           # acortar reviews a 200 palabras (500 es mucho para un RNN simple)

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test  = pad_sequences(X_test, maxlen=maxlen)

# 2. Definir el modelo
model = Sequential([
    Embedding(input_dim=max_features, output_dim=64, input_length=maxlen),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 3. Entrenar
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# 4. Evaluar
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
