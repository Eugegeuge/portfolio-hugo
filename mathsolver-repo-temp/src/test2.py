import tkinter as tk
from tkinter import Button, Label, filedialog
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt  # Para depuración y visualización

# Cargar el modelo entrenado
model_path = "svm_model.joblib"
svm_model = joblib.load(model_path)

# Etiquetas de las clases
class_names = ["dot", "minus", "plus", "slash", "x"]  # Cambia por los nombres reales

# Dimensiones de la imagen
IMAGE_SIZE = 28

def preprocess_image(img):
    """
    Procesa la imagen para centrar el contenido:
    1. Detecta bordes no vacíos.
    2. Recorta la región con contenido.
    3. Redimensiona a 28x28 píxeles.
    4. Invierte los colores.
    """
    # Convertir a array de NumPy
    img_array = np.array(img)

    # Detectar bordes no vacíos
    non_empty_columns = np.where(img_array < 255)[1]
    non_empty_rows = np.where(img_array < 255)[0]

    if len(non_empty_columns) == 0 or len(non_empty_rows) == 0:
        # Si no se detecta contenido, devolver una imagen en blanco
        return Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)

    crop_box = (min(non_empty_columns), min(non_empty_rows),
                max(non_empty_columns) + 1, max(non_empty_rows) + 1)

    # Recortar la región con contenido
    img_cropped = img.crop(crop_box)

    # Redimensionar a 28x28 píxeles
    img_resized = img_cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    # Invertir colores (fondo blanco y contenido negro)
    img_inverted = ImageOps.invert(img_resized)

    # Depuración: Mostrar la imagen invertida
    plt.imshow(img_resized, cmap='gray')
    plt.title("Imagen Invertida")
    plt.show()

    return img_resized

def predict_from_image():
    """
    Permite al usuario cargar una imagen desde una ruta, procesarla y realizar la predicción.
    """
    # Abrir un cuadro de diálogo para seleccionar la imagen
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not file_path:
        return  # Si no se selecciona ningún archivo, salir

    # Cargar y procesar la imagen
    img = Image.open(file_path).convert("L")  # Convertir a escala de grises
    img_processed = preprocess_image(img)

    # Aplanar y normalizar
    img_array = np.array(img_processed).reshape(1, -1) / 255.0  # Aplanar y normalizar

    # Realizar la predicción
    prediction = svm_model.predict(img_array)
    predicted_class = class_names[prediction[0]]

    # Mostrar el resultado
    result_label.config(text=f"Predicción: {predicted_class}")

# Crear la ventana principal
window = tk.Tk()
window.title("Reconocimiento de símbolos")

# Botón para predecir desde una imagen
load_image_button = Button(window, text="Cargar imagen y predecir", command=predict_from_image)
load_image_button.grid(row=0, column=0, columnspan=2, pady=10)

# Etiqueta para mostrar el resultado
result_label = Label(window, text="Predicción: ", font=("Arial", 16))
result_label.grid(row=1, column=0, columnspan=2)

# Iniciar la aplicación
window.mainloop()
