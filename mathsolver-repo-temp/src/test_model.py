from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Función para preprocesar la imagen con estiramiento horizontal y márgenes blancos
def preprocess_digit(img):
    """
    Preprocesa la imagen: estira ligeramente el dígito en el eje horizontal,
    agrega márgenes blancos y normaliza para el modelo.
    """
    # Asegurarse de que la imagen esté en escala de grises
    if len(img.shape) == 3:  # Si tiene 3 canales, convertirla a escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarizar para resaltar el dígito (negro sobre blanco)
    _, img_binaria = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Redimensionar a una altura fija de 20 píxeles con estiramiento horizontal
    h, w = img_binaria.shape
    aspect_ratio = w / h
    stretch_factor = 1.2  # Factor de estiramiento horizontal
    new_height = 20
    new_width = int(aspect_ratio * new_height * stretch_factor)
    img_resized = cv2.resize(img_binaria, (new_width, new_height))

    # Crear una imagen de 28x28 con fondo blanco y centrar el dígito
    padded_img = np.full((28, 28), 255, dtype=np.uint8)  # Fondo blanco
    x_offset = (28 - new_width) // 2  # Márgenes izquierdo y derecho
    y_offset = 4  # Márgenes superior e inferior (centrar verticalmente)
    if new_width > 28:  # Recortar si el estiramiento excede el ancho permitido
        new_width = 28
        img_resized = cv2.resize(img_resized, (new_width, new_height))
        x_offset = 0
    padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

    # Normalizar para el modelo
    img_normalized = (255 - padded_img) / 255.0  # Invertir colores para el modelo
    img_preprocessed = img_normalized.reshape(1, 28, 28, 1)  # Ajustar las dimensiones para el modelo

    return padded_img, img_preprocessed

# Ruta de la imagen de prueba
img_path = 'caracteres_separados/char_0.png'  # Cambia esto a la ruta de tu imagen
img = cv2.imread(img_path)

# Procesar la imagen
processed_img, preprocessed_img = preprocess_digit(img)

# Mostrar la imagen procesada antes de predecir
plt.figure(figsize=(6, 6))
plt.imshow(processed_img, cmap='gray')
plt.title("Imagen preprocesada con estiramiento horizontal")
plt.axis("off")
plt.show()

# Cargar el modelo entrenado
model = load_model('digit_recognition_model.keras')

# Realizar la predicción
prediction = model.predict(preprocessed_img)
digit = np.argmax(prediction)
print(f"El dígito reconocido es: {digit}")
