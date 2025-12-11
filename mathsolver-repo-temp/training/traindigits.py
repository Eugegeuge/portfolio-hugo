import cv2
import numpy as np
import os
import glob
import time

# Variable global para guardar los elementos de la imagen
vector_images = []
vector_colors = [] #vector booleano: True para rojo, False para negro

def extract_characters_from_image(img, output_dir):
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Eliminar todos los archivos en el directorio de salida
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            os.remove(f)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionamos
    gray = cv2.resize(gray, (500, 220))
    # tambien la imagen original
    img_resized = cv2.resize(img, (500, 220))

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    cv2.imwrite('imagen_gris.png', gray)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)

    altura, ancho = thresh.shape[0:2]
    # Limpiar bordes no deseados
    thresh[0:altura, 0:(ancho//40)] = 0
    thresh[0:altura, (ancho-ancho//40):ancho] = 0
    thresh[0:(altura//28), 0:ancho] = 0
    thresh[(altura-altura//28):altura, 0:ancho] = 0 
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(thresh)
    min_area = 50

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_area:
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(clean_mask, [contour], -1, (255), thickness=cv2.FILLED)
            else:
                cv2.drawContours(clean_mask, [contour], -1, (0), thickness=cv2.FILLED)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar los contornos de izquierda a derecha
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Definir rango para color rojo en HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    index = 0
    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        # Ignorar contornos pequeños que pueden ser ruido
        if w > 5 and h > 5:
            # Recortar el carácter de la imagen umbralizada
            char_img = clean_mask[y:y+h, x:x+w]

            # Recortar la región correspondiente en la imagen redimensionada original
            char_region = img_resized[y:y+h, x:x+w]

            # Convertir a HSV la región del carácter en la imagen original redimensionada
            hsv_char = cv2.cvtColor(char_region, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv_char, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_char, lower_red2, upper_red2)
            red_mask_char = cv2.bitwise_or(mask1, mask2)
            
            # Contar píxeles del carácter (blancos en char_img) y cuántos son rojos
            char_pixels = np.sum(char_img == 255)
            red_pixels = np.sum((char_img == 255) & (red_mask_char == 255))
            
            # Determinar si es rojo (más del 50% de píxeles del carácter son rojos)
            is_red = (red_pixels / char_pixels > 0.5) if char_pixels > 0 else False
            vector_colors.append(is_red)

            # Guardar la imagen del carácter
            char_filename = os.path.join(output_dir, f'char_{index}.png')
            cv2.imwrite(char_filename, char_img)
            index += 1
            vector_images.append(char_img)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta los mensajes de información y advertencias

    output_dir = 'caracteres_separados' #LAS IMAGENES FRAGMENTADAS SE GUARDAN EN ESTE DIRECTORIO

    # Solicitar la ruta de la imagen al usuario
    image_path = input("Introduce el path de la imagen: ")

    if not os.path.exists(image_path):
        print("La ruta de la imagen no es válida.")
        return

    # Leer la imagen desde el path proporcionado
    img = cv2.imread(image_path)

    if img is None:
        print("No se pudo cargar la imagen. Verifica el path.")
        return

    # Procesar la imagen cargada
    extract_characters_from_image(img, output_dir)
    print("Imagen procesada exitosamente.")
    print(f"Vector de colores (True para rojo, False para negro): {vector_colors}")

    # Parte comentada de la captura desde la cámara
    # cap = cv2.VideoCapture(0) 

    # if not cap.isOpened():
    #     print("No se puede abrir la cámara")
    #     exit()

    # # Tomamos la referencia de tiempo al iniciar la cámara
    # start_time = time.time()
    # capture_delay = 4  # Segundos antes de capturar automáticamente

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("No se puede recibir frame (stream end?). Saliendo ...")
    #         break

    #     # Obtener dimensiones del frame
    #     height, width = frame.shape[:2]

    #     # Definir coordenadas del rectángulo (centrado)
    #     rect_width = int(width * 0.6)
    #     rect_height = int(height * 0.3)
    #     rect_x = int((width - rect_width) / 2)
    #     rect_y = int((height - rect_height) / 2)
    #     rect_top_left = (rect_x, rect_y)
    #     rect_bottom_right = (rect_x + rect_width, rect_y + rect_height)

    #     # Dibujar el rectángulo en el frame (color verde, grosor 3)
    #     cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 3)

    #     # Añadir texto del título y mensaje de instrucción
    #     cv2.putText(frame, "Math Resolver", (50, 40), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 2)
    #     cv2.putText(frame, "Alinea tu ecuacion dentro del recuadro", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (61, 128, 0), 2)
    #     
    #     # Mostrar el frame
    #     cv2.imshow('Captura - Math Resolver', frame)

    #     # Comprobar si han pasado los 4 segundos
    #     if time.time() - start_time >= capture_delay:
    #         # Capturar la imagen dentro del rectángulo automáticamente
    #         roi = frame[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         # Procesar la imagen capturada
    #         extract_characters_from_image(roi, output_dir)
    #         print("Imagen capturada y procesada automáticamente.")
    #         print(f"Vector de colores (True para rojo, False para negro): {vector_colors}")
    #         break

    #     # Pulsar 'q' para abortar la captura
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

 
    # cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
