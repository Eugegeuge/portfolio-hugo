import os
import cv2
import numpy as np

def procesar_y_guardar_imagenes(input_dir, output_dir):
    """
    Procesa imágenes con canal alfa, convierte el fondo transparente a blanco,
    convierte a escala de grises, invierte los colores (negro → blanco) y guarda en una nueva carpeta.
    Se mantiene la estructura de directorios original.

    Args:
        input_dir (str): Ruta a la carpeta de imágenes original.
        output_dir (str): Ruta donde se guardarán las imágenes procesadas.
    """
    # Recorrer la estructura de directorios
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Procesar solo imágenes
                # Leer la imagen con canal alfa si existe
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Lee BGRA si existe

                if img is None:
                    print(f"Error al leer la imagen: {img_path}")
                    continue

                # Si la imagen tiene 4 canales (BGRA), manejar el canal alfa
                if len(img.shape) == 3 and img.shape[2] == 4:
                    # Separar los canales de color y alfa
                    bgr, alpha = img[:, :, :3], img[:, :, 3]

                    # Crear una imagen blanca donde el alfa es transparente
                    white_bg = np.ones_like(bgr, dtype=np.uint8) * 255  # Fondo blanco
                    img = np.where(alpha[:, :, None] == 0, white_bg, bgr)  # Píxeles transparentes → blanco

                # Convertir a escala de grises
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Invertir los colores
                img_invertida = cv2.bitwise_not(gray_img)

                # Crear la ruta de salida manteniendo la estructura de carpetas
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                # Guardar la imagen procesada
                output_path = os.path.join(output_folder, file)
                cv2.imwrite(output_path, img_invertida)
                print(f"Imagen guardada: {output_path}")

# ============================
# Configuración
# ============================
input_directory = "archive/dataset/"        # Ruta al dataset original
output_directory = "archive/dataset_inverted/"  # Nueva ruta para guardar las imágenes procesadas

# Ejecutar la función
procesar_y_guardar_imagenes(input_directory, output_directory)

print("Todas las imágenes han sido procesadas y guardadas correctamente.")
