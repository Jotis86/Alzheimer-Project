import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Definir las rutas de los archivos
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'alzheimer_model.h5')
train_data_file_path = os.path.join(base_dir, 'train.parquet')
test_data_file_path = os.path.join(base_dir, 'test.parquet')
ml_model_path = os.path.join(base_dir, 'best_model.pkl')
feature_selector_path = os.path.join(base_dir, 'feature_selector.pkl')

# Cargar el modelo de deep learning
dl_model = load_model(model_path)


# Definir las clases para Deep Learning y Machine Learning
dl_class_names = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
ml_class_names = ['Negative', 'Positive']

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

# Menú de navegación
menu = ["Deep Learning", "Machine Learning"]
choice = st.sidebar.selectbox("Select Model Type", menu)

if choice == "Deep Learning":
    st.header("Deep Learning Model")

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
        cols[i % 2].image(image, caption=f"Label: {dl_class_names[row['label']]}", use_column_width=True)

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
        prediction = dl_model.predict(processed_image)
        predicted_class = dl_class_names[np.argmax(prediction)]
        
        # Display the prediction
        st.subheader("Model Prediction")
        st.write(f"The selected image has been classified as: **{predicted_class}**")
        
        # Display the probability of each class
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(dl_class_names):
            st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

elif choice == "Machine Learning":
    st.header("Machine Learning Model")

    # Application description
    st.markdown("""
    This application uses a machine learning model to detect the presence of Alzheimer's disease based on various features.
    You can manually input the values for the selected features, and the model will predict whether the data shows signs of Alzheimer's.
    """)

    # Cargar los datos de entrenamiento
    data = pd.read_csv('alzheimers_disease_data.csv')

    # Seleccionar características y variable objetivo
    selected_features = ['EducationLevel', 'DietQuality', 'SleepQuality',
                         'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Hypertension',
                         'SystolicBP', 'DiastolicBP', 'CholesterolHDL',
                         'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
                         'MemoryComplaints', 'BehavioralProblems', 'ADL']
    x = data[selected_features]
    y = data["Diagnosis"]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    # Definir el modelo
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

    # Entrenar el modelo
    model.fit(x_train, y_train)

    # Crear un formulario para la entrada de datos
    st.subheader("Input Feature Values")
    input_data = {}
    unique_values = {
        'EducationLevel': list(range(0, 4)),
        'DietQuality': list(range(0, 9)),
        'SleepQuality': list(range(5, 11)),
        'FamilyHistoryAlzheimers': [0, 1],
        'CardiovascularDisease': [0, 1],
        'Hypertension': [0, 1],
        'SystolicBP': list(range(90, 180)),
        'DiastolicBP': list(range(60, 120)),
        'CholesterolHDL': list(range(30, 100)),
        'CholesterolTriglycerides': list(range(80, 300)),
        'MMSE': list(range(4, 22)),
        'FunctionalAssessment': list(range(1, 8)),
        'MemoryComplaints': [0, 1],
        'BehavioralProblems': [0, 1],
        'ADL': list(range(1, 9))
    }
    
    for feature in selected_features:
        if feature in unique_values:
            input_data[feature] = st.selectbox(f"Select value for {feature}", unique_values[feature])
        else:
            input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

    # Convertir los datos de entrada en un DataFrame
    input_df = pd.DataFrame([input_data])

    # Verificar los datos de entrada
    st.write("Input Data:")
    st.write(input_df)

    # Hacer la predicción
    if st.button("Predict"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Mostrar la predicción
        st.subheader("Model Prediction")
        predicted_class = ml_class_names[prediction[0]]
        st.write(f"The input data has been classified as: **{predicted_class}**")

        # Mostrar la probabilidad de cada clase
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(ml_class_names):
            st.write(f"{class_name}: {prediction_proba[0][i]*100:.2f}%")

        # Verificar si el modelo está funcionando correctamente
        st.write("Model Check:")
        st.write(f"Prediction: {prediction}")
        st.write(f"Prediction Probabilities: {prediction_proba}")

# Footer
st.markdown("""
---
**Note:** This application is for educational purposes only and should not be used for real medical diagnoses.
""")