import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Configuración del dataset
# ============================
dataset_path = "archive/dataset_inverted/"

# Generador de datos sin aumentación
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalizar valores de píxeles (0-255 -> 0-1)
    validation_split=0.2  # Reservar 20% para validación
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(28, 28),  # Redimensionar imágenes a 28x28
    color_mode="grayscale",  # Convertir imágenes a escala de grises
    batch_size=32,
    class_mode="sparse",  # Etiquetas numéricas
    subset="training"  # Subconjunto de entrenamiento
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    class_mode="sparse",
    subset="validation"  # Subconjunto de validación
)

# ============================
# Creación del modelo
# ============================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Regularización para evitar sobreajuste
    layers.Dense(len(train_data.class_indices), activation='softmax')  # Número de clases dinámico
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================
# Entrenamiento del modelo
# ============================
print("Entrenando el modelo...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ============================
# Evaluación y visualización
# ============================

# Gráfica de entrenamiento
def plot_training_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Entrenamiento')
    plt.plot(epochs, history.history['val_accuracy'], label='Validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Entrenamiento')
    plt.plot(epochs, history.history['val_loss'], label='Validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()

plot_training_history(history)

# Matriz de confusión
def evaluate_and_plot_confusion_matrix(model, val_data):
    # Obtener todas las imágenes y etiquetas de validación
    val_images, val_labels = next(val_data)
    val_images = np.concatenate([val_images for _ in range(len(val_data))])
    val_labels = np.concatenate([val_labels for _ in range(len(val_data))])

    # Predicciones
    val_preds = np.argmax(model.predict(val_images), axis=1)

    # Matriz de confusión
    cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_data.class_indices.keys()))
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title('Matriz de Confusión (Validación)')
    plt.savefig("confusion_matrix.png")
    plt.close()

evaluate_and_plot_confusion_matrix(model, val_data)

# Ejemplos de predicciones
def plot_predictions(model, val_data):
    val_images, val_labels = next(val_data)
    val_images = np.concatenate([val_images for _ in range(len(val_data))])
    val_labels = np.concatenate([val_labels for _ in range(len(val_data))])

    val_preds = np.argmax(model.predict(val_images), axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(val_images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {list(val_data.class_indices.keys())[val_preds[i]]}\n(True: {list(val_data.class_indices.keys())[int(val_labels[i])]} )')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("predictions_examples.png")
    plt.close()

plot_predictions(model, val_data)

print("Gráficas generadas y guardadas: 'training_history.png', 'confusion_matrix.png', 'predictions_examples.png'")
