import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Cargar el modelo preentrenado
model_best = load_model('emotion_model_v1.h5', compile=False)

# Compilar el modelo después de cargarlo
model_best.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Clases de las emociones
class_names = ['Enojado', 'Disgustado', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

# Cargar el clasificador de rostros preentrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para procesar la transmisión de la cámara
def process_frame():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_image = cv2.resize(face_roi, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = np.vstack([face_image])
            predictions = model_best.predict(face_image)
            emotion_label = class_names[np.argmax(predictions)]
            cv2.putText(frame, f'Emoción: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_label.config(image=photo)
        video_label.image = photo
    if is_processing:
        video_label.after(10, process_frame)

# Funciones para iniciar y detener la detección de emociones
def start_detection():
    global is_processing
    is_processing = True
    process_frame()

def stop_detection():
    global is_processing
    is_processing = False

# Crear la ventana principal
root = tk.Tk()
root.title("Detección de Emociones")

# Crear un contenedor para la transmisión de la cámara
video_frame = tk.Frame(root)
video_frame.pack(padx=10, pady=10)

# Etiqueta para mostrar la transmisión de la cámara
video_label = tk.Label(video_frame)
video_label.pack()

# Botones para iniciar y detener la detección de emociones
button_frame = tk.Frame(root)
button_frame.pack(padx=10, pady=5)

start_button = tk.Button(button_frame, text="Iniciar Detección", command=start_detection)
start_button.grid(row=0, column=0, padx=5)

stop_button = tk.Button(button_frame, text="Detener Detección", command=stop_detection)
stop_button.grid(row=0, column=1, padx=5)

# Iniciar la captura de la cámara web
cap = cv2.VideoCapture(0)

# Bandera para indicar si se está procesando la transmisión de la cámara
is_processing = False

root.mainloop()

# Liberar la cámara web
cap.release()
