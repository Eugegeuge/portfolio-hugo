import cv2
import numpy as np
import matplotlib.pyplot as plt

def centrar_digito_ajustado(imagen):
    """
    Ajusta y centra un dígito en una imagen binaria con bordes complejos.

    Args:
        imagen (numpy.ndarray): Imagen en escala de grises, con un dígito sobre fondo blanco.

    Returns:
        numpy.ndarray: Imagen con el dígito centrado.
    """
    # Binarizar la imagen (invertir para dígito blanco sobre fondo negro)
    _, imagen_binaria = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY_INV)

    # Encontrar los contornos del dígito
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        raise ValueError("No se encontraron dígitos en la imagen.")

    # Encontrar el rectángulo delimitador que envuelve todos los contornos
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)

    # Recortar la región que contiene todos los contornos
    digito_recortado = imagen_binaria[y_min:y_max, x_min:x_max]

    # Calcular el tamaño del lienzo cuadrado
    h, w = digito_recortado.shape
    tamaño_lienzo = max(w, h)

    # Crear una imagen cuadrada con fondo negro
    lienzo = np.zeros((tamaño_lienzo, tamaño_lienzo), dtype=np.uint8)

    # Calcular las coordenadas para centrar el dígito
    offset_x = (tamaño_lienzo - w) // 2
    offset_y = (tamaño_lienzo - h) // 2

    # Colocar el dígito centrado en el lienzo
    lienzo[offset_y:offset_y+h, offset_x:offset_x+w] = digito_recortado

    # Restaurar colores originales (dígito negro sobre fondo blanco)
    imagen_centrada = cv2.bitwise_not(lienzo)

    return imagen_centrada

# Cargar la imagen
ruta_imagen = '0.png'  # Cambia por la ruta de tu archivo
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Aplicar la función ajustada para centrar el dígito
imagen_centrada = centrar_digito_ajustado(imagen)

# Mostrar la imagen original y centrada
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagen Centrada Ajustada")
plt.imshow(imagen_centrada, cmap='gray')
plt.axis('off')

plt.show()
