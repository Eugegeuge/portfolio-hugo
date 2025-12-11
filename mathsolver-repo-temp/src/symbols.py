from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import joblib  # Para guardar y cargar el modelo

# Configuración del generador de datos
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.4  # 20% para validación, 20% para test
)

# Dataset
dataset_path = r"C:\Users\hugos\Documents\UNIVERSITY\TERCERO\Visión por computador\digitRecognition\bhmsds-master\bhmsds-master\organized_symbols"

# Dividir los datos en entrenamiento, validación y test
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=False
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Convertir los datos en formato adecuado para SVM
def generator_to_data(generator):
    """
    Convierte un generador de datos de Keras en arrays NumPy de imágenes y etiquetas.
    """
    images, labels = [], []
    for i in range(len(generator)):
        batch_images, batch_labels = next(generator)
        images.append(batch_images)
        labels.append(batch_labels)
    images = np.vstack(images)
    labels = np.argmax(np.vstack(labels), axis=1)  # Convertir etiquetas "one-hot" a índices
    return images, labels

# Obtener datos de entrenamiento, validación y test
train_images, train_labels = generator_to_data(train_data)
val_images, val_labels = generator_to_data(val_data)
test_images, test_labels = generator_to_data(test_data)

# Aplanar las imágenes para SVM
train_images_flat = train_images.reshape(train_images.shape[0], -1)
val_images_flat = val_images.reshape(val_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Entrenar SVM
print("Entrenando el modelo SVM...")
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(train_images_flat, train_labels)

# Guardar el modelo entrenado
model_path = "svm_model2.joblib"
joblib.dump(svm, model_path)
print(f"Modelo SVM guardado en: {model_path}")

# Cargar el modelo guardado (opcional)
loaded_svm = joblib.load(model_path)
print("Modelo SVM cargado con éxito.")

# Evaluar el modelo en validación
val_preds = loaded_svm.predict(val_images_flat)

# Reporte de clasificación en validación
class_names = list(train_data.class_indices.keys())
print("\nReporte de clasificación (Validación):")
print(classification_report(val_labels, val_preds, target_names=class_names))

# Evaluar el modelo en test
test_preds = loaded_svm.predict(test_images_flat)

# Reporte de clasificación en test
print("\nReporte de clasificación (Test):")
print(classification_report(test_labels, test_preds, target_names=class_names))

# Matriz de confusión en test
cm = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title('Matriz de confusión (Test)')
plt.show()
