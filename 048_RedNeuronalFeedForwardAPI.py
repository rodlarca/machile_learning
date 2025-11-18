from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, concatenate

# Definir las entradas
input_a = Input(shape=(28*28,), name='input_a')
input_b = Input(shape=(28*28,), name='input_b')

# Crear un bloque de capas
shared_dense = Dense(128, activation='relu')

# Aplicar el bloque a ambas entradas
processed_a = shared_dense(input_a)
processed_b = shared_dense(input_b)

# Concatenar las salidas
concatenated = concatenate([processed_a, processed_b], axis=-1)

# Agregar capas adicionales
output = Dense(10, activation='softmax')(concatenated)

# Crear el modelo
model = Model(inputs=[input_a, input_b], outputs=output)

# Compilar y entrenar el modelo (usando datos de ejemplo)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])