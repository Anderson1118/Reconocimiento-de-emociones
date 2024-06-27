import cv2
import numpy as np
from tkinter import Tk, filedialog
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt

# Cargar el modelo preentrenado
model_best = load_model('emotion_model.h5', compile=False) # Ajusta la ruta del modelo y carga sin compilar el modelo

# Compilar el modelo después de cargarlo
model_best.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Clases de las emociones
class_names = ['Enojado', 'Disgustado', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

# Cargar el clasificador de rostros preentrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises para la detección de rostros
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        # Extraer la región de la cara
        face_roi = img[y:y + h, x:x + w]

        # Cambiar el tamaño de la imagen de la cara al tamaño de entrada requerido para el modelo
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        # Predecir la emoción utilizando el modelo cargado
        predictions = model_best.predict(face_image)
        emotion_probabilities = predictions[0]

        # Obtener la emoción predominante y su probabilidad
        predominant_emotion_index = np.argmax(emotion_probabilities)
        predominant_emotion = class_names[predominant_emotion_index]
        predominant_emotion_probability = emotion_probabilities[predominant_emotion_index]

        # Mostrar la imagen con la emoción predominante resaltada y el gráfico de barras
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Mostrar la imagen con la detección de emociones
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.axis('off')
        ax1.set_title(f'Emoción Detectada: {predominant_emotion}')
        
        # Resaltar la región del rostro con un rectángulo
        ax1.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
        # Mostrar el gráfico de barras de las probabilidades de cada emoción
        ax2.bar(class_names, emotion_probabilities, color='blue')
        ax2.set_title('Probabilidades de Emociones')
        ax2.set_xlabel('Emoción')
        ax2.set_ylabel('Probabilidad')


        plt.tight_layout()
        plt.show()

def select_image():
    while True:
        # Crear una ventana Tkinter
        root = Tk()
        root.withdraw()  # Ocultar la ventana principal de Tkinter

        # Mostrar el cuadro de diálogo de selección de archivo y obtener la ruta de la imagen seleccionada
        file_path = filedialog.askopenfilename()

        # Comprobar si se seleccionó un archivo
        if file_path:
            detect_emotion(file_path)
        else:
            break

# Llamar a la función select_image para permitir al usuario seleccionar la imagen
select_image()
