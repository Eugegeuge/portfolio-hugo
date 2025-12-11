# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar el dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar los datos (de 0-255 a 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Redimensionar los datos para que tengan una dimensión de canal (necesario para CNN)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Crear el modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 clases para los dígitos del 0 al 9
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
print("Entrenando el modelo, esto puede tomar unos minutos...")
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Guardar el modelo en el formato nativo de Keras
model.save('digit_recognition_model.keras')
print("Modelo guardado como 'digit_recognition_model.keras'")
