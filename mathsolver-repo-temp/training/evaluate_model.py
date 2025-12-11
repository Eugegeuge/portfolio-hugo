# evaluate_model.py
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Cargar el modelo entrenado
model = load_model('digit_recognition_model.keras')

# Cargar los datos MNIST
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocesar los datos de prueba
x_test = x_test / 255.0  # Normalizar
x_test = x_test.reshape(-1, 28, 28, 1)  # Agregar dimensión de canal

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Precisión en el conjunto de prueba: {test_acc * 100:.2f}%")
