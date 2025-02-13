import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import os

# Definir las rutas de los archivos
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'alzheimer_model.h5')
train_data_file_path = os.path.join(base_dir, './MRI Dataset/Data/train.parquet')
test_data_file_path = os.path.join(base_dir, './MRI Dataset/Data/test.parquet')

# Cargar el modelo entrenado
model = load_model(model_path)

# Definir las clases
class_names = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

# Función para convertir bytes a imagen
def bytes_to_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Función para preprocesar la imagen
def preprocess_image(image):
    image = np.array(image)
    if image.ndim == 2:  # Si la imagen es en blanco y negro, convertir a 3 canales
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (128, 128))  # Redimensionar la imagen
    image = image / 255.0  # Normalizar la imagen
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión para el batch
    return image

# Cargar los datos
train_data = pd.read_parquet(train_data_file_path)
test_data = pd.read_parquet(test_data_file_path)

# Título de la aplicación
st.title("Alzheimer Detection from MRI Scans")

# Seleccionar una imagen del conjunto de datos
image_selection = st.selectbox("Select an MRI scan image from the dataset:", test_data.index)

if image_selection is not None:
    # Obtener la imagen seleccionada
    image_bytes = train_data.loc[image_selection, 'image']['bytes']
    image = bytes_to_image(image_bytes)
    
    # Mostrar la imagen seleccionada
    st.image(image, caption='Selected MRI Scan.', use_container_width=True)
    
    # Preprocesar la imagen
    processed_image = preprocess_image(image)
    
    # Hacer la predicción
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    
    # Mostrar la predicción
    st.write(f"Prediction: {predicted_class}")