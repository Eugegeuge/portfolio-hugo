import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

def preprocess_image_fix_x(img_path):
    """
    Preprocesa una imagen para que sea compatible con el modelo SVM.
    - Mantiene las proporciones del símbolo al redimensionarlo.
    - Si el símbolo es más alto que ancho, ajusta dinámicamente su ancho para evitar que se comprima demasiado.
    - Centra el símbolo en un lienzo de 28x28 píxeles con un fondo blanco.
    """
    # Leer la imagen en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en {img_path}")

    # Invertir colores para garantizar que el fondo sea blanco y el símbolo negro
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Detectar contornos para recortar la región del símbolo
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar el rectángulo delimitador que rodea todos los contornos
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)

        cropped_img = thresh[y_min:y_max, x_min:x_max]
    else:
        cropped_img = thresh  # Si no hay contornos, usar la imagen binarizada

    # Obtener las dimensiones originales del símbolo
    h, w = cropped_img.shape

    # Ajustar proporciones para evitar compresión excesiva
    if h > w:  # Si el símbolo es más alto que ancho
        new_h = 20
        new_w = max(int(w * (20 / h)), 10)  # Asegurarse de que el ancho sea razonable
    else:
        new_w = 20
        new_h = max(int(h * (20 / w)), 10)  # Asegurarse de que la altura sea razonable

    img_resized = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Crear un lienzo blanco de 28x28
    img_with_borders = np.full((28, 28), 255, dtype=np.uint8)

    # Calcular las coordenadas para centrar el símbolo en el lienzo
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    img_with_borders[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    return img_with_borders

# Ruta de la imagen del dot
img_dot_path = "caracteres_separados/char_2.png"

# Procesar la imagen del dot
processed_img_dot = preprocess_image_fix_x(img_dot_path)

# Mostrar la imagen procesada
plt.figure(figsize=(6, 6))
plt.title("Imagen Procesada")
plt.imshow(processed_img_dot, cmap='gray')
plt.axis("off")
plt.show()

# Cargar el modelo SVM
model = joblib.load('svm_model.joblib')

# Etiquetas de las clases
class_names = ["dot", "minus", "plus", "slash", "x"]

# Realizar la predicción
# Aplanar la imagen procesada y normalizar
preprocessed_img_flattened = processed_img_dot.flatten() / 255.0
prediction = model.predict([preprocessed_img_flattened])
predicted_class = class_names[prediction[0]]

# Mostrar la predicción
print(f"El símbolo reconocido es: {predicted_class}")
