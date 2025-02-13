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

# Application description
st.markdown("""
This application uses a deep learning model to detect the presence of Alzheimer's disease in MRI brain scan images.
You can select an image from the test dataset, and the model will predict whether the image shows signs of Alzheimer's.
""")

# Display some sample images from the dataset
st.header("Sample Images from the Dataset")
sample_images = train_data.sample(4)
cols = st.columns(2)
for i, (index, row) in enumerate(sample_images.iterrows()):
    image = bytes_to_image(row['image']['bytes'])
    cols[i % 2].image(image, caption=f"Label: {class_names[row['label']]}", use_container_width=True)

# Select an image from the test dataset
st.header("Select an Image for Prediction")
image_selection = st.selectbox("Select an MRI scan image from the test dataset:", test_data.index)

if image_selection is not None:
    # Get the selected image
    image_bytes = test_data.loc[image_selection, 'image']['bytes']
    image = bytes_to_image(image_bytes)
    
    # Display the selected image
    st.image(image, caption='Selected MRI Scan.', use_container_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make the prediction
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    
    # Display the prediction
    st.subheader("Model Prediction")
    st.write(f"The selected image has been classified as: **{predicted_class}**")
    
    # Display the probability of each class
    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")