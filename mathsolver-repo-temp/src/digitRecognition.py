from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
import joblib
import sympy as sp 

# Variable global para guardar los elementos de la imagen
vector_images = []
vector_colors = [] #vector booleano: True para rojo, False para negro

def preprocesar_ecuacion(ecuacion):     # 3x debe pasarse como 3*x, con los parentesis igual

    i = 0
    aux = ""
    while(i < len(ecuacion)):
        if (ecuacion[i] in ('x(') and ecuacion[i-1].isdigit()):
            aux += '*'
        aux += ecuacion[i]
        i += 1
    return ''.join(aux)


def cambiar_signos_der (ecuacion):

    changed_signos = ""
    in_parentesis = False
    i = 0

    while(i < len(ecuacion)):

        if (ecuacion[i] == '('):    # Se abre parentesis
            in_parentesis = True
            changed_signos += '('
        
        elif (ecuacion[i] == ')'):    # Se cierra el parentesis
            in_parentesis = False
            changed_signos += ')'
        
        else:
            if (not in_parentesis):
                if (ecuacion[i] == '+'):
                    changed_signos += '-'
                elif (ecuacion[i] == '-'):
                    changed_signos += '+'
                else:
                    changed_signos += ecuacion[i]
            else: 
                changed_signos += ecuacion[i]
        
        i += 1
    return ''.join(changed_signos)


def ecuacion_izq(expresion):

    # Primero sustituimos los espacios (si los hubiera) en blanco de ambos lados (esto no deberia pasar pq daria problemas entre otras cosas)
    expresion = expresion.replace(" ","")

    # Preprocesamos las multiplicaciones de la ecuacion
    expresion = preprocesar_ecuacion(expresion)
    #print(f'Ecuacion con multiplicaciones: {expresion}')

    # Dividimos la ecuacion en dos partes separadas por el '='

    # Si no hay un igual hace la operacion obtenida
    if '=' not in expresion:
        print("OPCION DETECTADA: OPERACION MATEMÁTICA")
        sol = eval(expresion)
        print(f"Solucion = {sol}")

    else:
        print("OPCION DETECTADA: ECUACION")
                                                        
        izq, der = expresion.split('=')                  # Se separa la ecuacion por el = y se trabaja independientemente con ambas

        # Comprobamos si el primer número de la der es positivo o negativo para ponerle el '+' si es positivo
        if der and not der[0] in ('+', '-'):
            der = '+' + der

        # Cambiamos los signos de la derecha ('+' --> '-' y viceversa)
        der = cambiar_signos_der(der)
        #print(f'Derecha cambiada de signo: {der}')

        # Juntamos la ecuacion como si estuviera igualada a 0
        eq = f"{izq}{der}"
        # print("Ecuacion inicial igualada a 0:")
        # print(f"{eq} = 0")
        sol = res_eq(eq)
        print(f"Solucion = {sol}")

    
def res_eq (ecuacion):
    # se define la variable simbólica
    x = sp.symbols('x')

    # convertir el string en una expresion simbolcia
    ecuacion = sp.sympify(ecuacion)

    # Se resuelve
    sol = sp.solve(ecuacion, x)

    if sol:
        return sol
    else:
        return "NO SOL"
    
# -----------------------------------

def preprocess_symbol(img):
    """
    Preprocesa una imagen para que sea compatible con el modelo SVM.
    - Mantiene las proporciones del símbolo al redimensionarlo.
    - Si el símbolo es más alto que ancho, ajusta dinámicamente su ancho para evitar que se comprima demasiado.
    - Centra el símbolo en un lienzo de 28x28 píxeles con un fondo blanco.
    """
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
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta los mensajes de información y advertencias


    output_dir = 'caracteres_separados' #LAS IMAGENES FRAGMENTADAS SE GUARDAN EN ESTE DIRECTORIO

    # Solicitar la ruta de la imagen al usuario
    img_path = input("Introduce el path de la imagen: ")

    if not os.path.exists(img_path):
        print("La ruta de la imagen no es válida.")
        return

    # Leer la imagen desde el path proporcionado
    img = cv2.imread(img_path)

    if img is None:
        print("No se pudo cargar la imagen. Verifica el path.")
        return

    # cap = cv2.VideoCapture(0) 

    # if not cap.isOpened():
    #     print("No se puede abrir la cámara")
    #     exit()

    # # Tomamos la referencia de tiempo al iniciar la cámara
    # start_time = time.time()
    # capture_delay = 7  # Segundos antes de capturar automáticamente

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
        
    #     # Mostrar el frame
    #     cv2.imshow('Captura - Math Resolver', frame)

    #     # Comprobar si han pasado los 4 segundos
    #     if time.time() - start_time >= capture_delay:
    #         # Capturar la imagen dentro del rectángulo automáticamente
    #         img = frame[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         # Procesar la imagen capturada
    #         print("Imagen capturada y procesada automáticamente.")
    #         print(f"Vector de colores (True para rojo, False para negro): {vector_colors}")
    #         break

    #     # Pulsar 'q' para abortar la captura
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #        exit()

 
    # cap.release()

    # Procesar la imagen cargada
    extract_characters_from_image(img, output_dir)
    print("Imagen procesada exitosamente.")
    print(f"Vector de colores (True para rojo, False para negro): {vector_colors}")

    # Cargar los modelos
    digit_model = load_model('digit_recognition_model.keras')
    symbol_model = joblib.load('svm_model.joblib')

    # Etiquetas de las clases para los símbolos
    class_names = {"dot": ".", "minus": "-", "plus": "+", "slash": "/", "x": "x"}

    equation = ""

    for i, (char_img, is_red) in enumerate(zip(vector_images, vector_colors)):
        if is_red:
            # Preprocesar como símbolo
            preprocessed_img = preprocess_symbol(char_img)
            
            # Aplanar la imagen para el modelo SVM y normalizar
            preprocessed_img_flattened = preprocessed_img.flatten() / 255.0
            
            # Realizar predicción con el modelo de símbolos
            prediction = symbol_model.predict([preprocessed_img_flattened])
            result = class_names[list(class_names.keys())[prediction[0]]]
        else:
            # Preprocesar como dígito
            padded_img, preprocessed_img = preprocess_digit(char_img)
            
            # Realizar predicción con el modelo de dígitos
            prediction = digit_model.predict(preprocessed_img)
            result = str(np.argmax(prediction))
        
        # Agregar el resultado a la ecuación
        equation += result

    for i in range(len(equation) - 1):  # Asegúrate de no salir del rango
        if equation[i] == "-" and equation[i+1] == "-":
            equation = equation[:i] + "=" + equation[i+2:]  # Reemplazar '--' por '='
            break  # Sal del bucle si solo deseas reemplazar el primer caso

    equation = equation.replace(".", "*")

    # Mostrar la ecuación completa
    print("Ecuación formada:", equation)

    ecuacion_izq(equation)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
