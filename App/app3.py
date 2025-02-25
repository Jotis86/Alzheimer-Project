import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import os
import joblib
import cohere
from dotenv import load_dotenv


# Cargar las variables de entorno
load_dotenv()

# Obtener la clave API de Cohere desde la variable de entorno
cohere_api_key = os.getenv('COHERE_API_KEY')

# Inicializar el cliente de Cohere
co = cohere.Client(cohere_api_key)

# Función para obtener la respuesta del chat bot
def get_cohere_response(prompt):
    response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=["--"]
    )
    response_text = response.generations[0].text.strip()

    # Verificar si la respuesta está relacionada con el Alzheimer
    if "Alzheimer" not in response_text and "dementia" not in response_text:
        return "Sorry, I can only answer questions related to Alzheimer's disease."
    
    return response_text

# Prompt específico para el chat bot de Alzheimer
def generate_prompt(user_input):
    base_prompt = """
    You are an AI assistant specialized in Alzheimer's disease. Answer the following questions with accurate and helpful information about Alzheimer's disease.
    Provide concise and summarized responses within the token limit.
    If the question is not related to Alzheimer's disease, respond with "Sorry, I can only answer questions related to Alzheimer's disease."
    """
    return f"{base_prompt}\n\nUser: {user_input}\nAI:"

# Verificar el directorio actual de trabajo
current_dir = os.getcwd()
print("Directorio actual:", current_dir)
print("Contenido del directorio 'App/':", os.listdir("App"))

# Definir las rutas relativas dentro de la carpeta "App"
model_path = os.path.join('App', 'alzheimer_model.h5')
train_data_file_path = os.path.join('App', 'train.parquet')
test_data_file_path = os.path.join('App', 'test.parquet')
ml_model_path = os.path.join('App', 'best_random_forest_model.pkl')
feature_selector_path = os.path.join('App', 'feature_selector.pkl')
navigation_image_path = os.path.join('App', 'Image_3.jpeg')
home_image_path = os.path.join('App', 'image_2.jpeg')
ml_report_path = os.path.join('App', 'ML_Report.pdf')
dl_report_path = os.path.join('App', 'DL_Report.pdf')



# Cargar el modelo de deep learning
dl_model = load_model(model_path)

# Cargar el modelo de machine learning y el selector de características
ml_model = joblib.load(ml_model_path)
feature_selector = joblib.load(feature_selector_path)



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
st.title("🧠 Alzheimer AI: Detect & Support")

# Menú de navegación
st.sidebar.image(navigation_image_path, use_container_width=True)
menu = ["Home", "Power BI", "Machine Learning", "Deep Learning", "Chat Bot", "Other Resources"]
choice = st.sidebar.radio("Navigate", menu)

# Botón de enlace a GitHub
st.sidebar.markdown("""
<a href="https://github.com/Jotis86/Alzheimer-Project" target="_blank">
    <button style="background-color: #007BFF; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">
        GitHub Repository
    </button>
</a>
""", unsafe_allow_html=True)

# Tabla con los nombres de los miembros del proyecto
st.sidebar.markdown("## Project Members")
st.sidebar.markdown("""
| Name           | Role         |
|----------------|--------------------------------|
| [Juan Duran](https://github.com/Jotis86)     | Data Analyst 📊     |
| [Andrea Lafarga](https://github.com/AndreaLaHe) | Data Engineer 🛠️   |
""", unsafe_allow_html=True)


# Imagen principal en todas las pestañas
st.image(home_image_path, use_container_width=True)

if choice == "Home":
    st.header("Welcome to Alzheimer AI: Detection and Support")
    st.markdown("""
    This project aims to assist in the early detection of Alzheimer's disease using advanced machine learning and deep learning techniques. 🧠💡
    
    ### Objectives of the Project:
    - **Early Detection**: Utilize state-of-the-art models to detect Alzheimer's disease at an early stage. 🕵️‍♂️
    - **Accessibility**: Provide an easy-to-use interface for healthcare professionals and researchers. 🌐
    - **Education**: Raise awareness and educate users about Alzheimer's disease and its early signs. 📚

    ### Features of This Application:
    - **Power BI Dashboards**: Visualize data and gain insights through interactive dashboards. 📈
    - **Machine Learning Model**: Input various clinical and demographic features to get a prediction. 📊
    - **Deep Learning Model**: Predict Alzheimer’s from MRI scans. 🖼️🔍
    - **Chat Bot**: Get answers to your questions and support related to Alzheimer's disease. 💬
    - **Additional Resources**: Access research papers, support groups, and more. 📑

    Explore the different sections to learn more about the models, interact with a chat bot, and access additional resources. 🚀
    """)

elif choice == "Power BI":
    st.header("Power BI")
    st.markdown("""
    This section will feature Power BI dashboards and reports to provide insights and visualizations related to Alzheimer's disease.
    """)

elif choice == "Machine Learning":
    st.header("Machine Learning Model")

    # Application description
    st.markdown("""
    This dataset contains extensive health information for 2,149 patients. The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, symptoms, and a diagnosis of Alzheimer's Disease.

    This application uses a machine learning model to detect the presence of Alzheimer's disease based on various features.
    You can manually input the values for the selected features, and the model will predict whether the data shows signs of Alzheimer's.
    """)

    st.markdown("""
    **EducationLevel**: The education level of the patients, coded as follows:
    - 0: None
    - 1: High School
    - 2: Bachelor's
    - 3: Higher

    **SleepQuality**: Sleep quality score, ranging from 1 to 10.

    **SystolicBP**: Systolic blood pressure, ranging from 90 to 180 mmHg.

    **DiastolicBP**: Diastolic blood pressure, ranging from 60 to 120 mmHg.

    **CholesterolHDL**: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL.

    **MMSE**: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.

    **FunctionalAssessment**: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.

    **MemoryComplaints**: Presence of memory complaints, where 0 indicates No and 1 indicates Yes.

    **BehavioralProblems**: Presence of behavioral problems, where 0 indicates No and 1 indicates Yes.

    **ADL**: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.
    """)

    # Inicializar la lista de predicciones si no existe
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []

    # Definir todas las características esperadas
    all_features = [
        'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 'AlcoholConsumption',
        'PhysicalActivity', 'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
        'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
        'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
        'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
    ]

    # Definir las características seleccionadas por el modelo
    selected_features = [
        'EducationLevel', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
        'CholesterolHDL', 'MMSE', 'FunctionalAssessment', 'MemoryComplaints',
        'BehavioralProblems', 'ADL'
    ]

    # Definir los rangos mínimos y máximos para cada característica
    feature_ranges = {
        'EducationLevel': (0, 3),
        'SleepQuality': (1, 10),
        'SystolicBP': (90, 180),
        'DiastolicBP': (60, 120),
        'CholesterolHDL': (20, 100),
        'MMSE': (0, 30),
        'FunctionalAssessment': (0, 10),
        'MemoryComplaints': (0, 1),
        'BehavioralProblems': (0, 1),
        'ADL': (0, 10)
    }

    # Crear un formulario para la entrada de datos solo con las características seleccionadas
    st.subheader("Input Feature Values")
    input_data = {}
    cols = st.columns(2)  # Crear dos columnas
    for i, feature in enumerate(selected_features):
        min_val, max_val = feature_ranges[feature]
        with cols[i % 2]:  # Alternar entre las dos columnas
            if feature in ['MemoryComplaints', 'BehavioralProblems']:
                input_data[feature] = st.selectbox(f"Select value for {feature}", [0, 1])
            else:
                input_data[feature] = st.number_input(f"Enter value for {feature}", min_value=min_val, max_value=max_val, value=min_val)

    # Crear un DataFrame con todas las características esperadas, rellenando con valores predeterminados
    input_full_data = {feature: 0 for feature in all_features}
    input_full_data.update(input_data)
    input_df = pd.DataFrame([input_full_data])

    # Seleccionar las características principales
    input_selected = feature_selector.transform(input_df)

    # Verificar los datos de entrada
    st.write("Input Data:")
    st.write(input_df[selected_features])

    # Hacer la predicción
    if st.button("Predict"):
        prediction = ml_model.named_steps['classifier'].predict(input_selected)
        prediction_proba = ml_model.named_steps['classifier'].predict_proba(input_selected)

        # Guardar la predicción
        st.session_state['predictions'].append({
            'Input Data': input_df[selected_features].to_dict(orient='records')[0],
            'Prediction': ml_class_names[prediction[0]],
            'Probabilities': {class_name: prediction_proba[0][i] for i, class_name in enumerate(ml_class_names)}
        })

        # Mostrar la predicción
        st.subheader("Model Prediction")
        predicted_class = ml_class_names[prediction[0]]
        if predicted_class == 'Positive':
            st.write(f"The input data has been classified as: **{predicted_class}** 🟢")
        else:
            st.write(f"The input data has been classified as: **{predicted_class}** 🔴")
        
        # Mostrar la probabilidad de cada clase
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(ml_class_names):
            st.write(f"{class_name}: {prediction_proba[0][i]*100:.2f}%")

    # Generar el archivo CSV de predicciones
    predictions_df = pd.DataFrame(st.session_state['predictions'])
    csv = predictions_df.to_csv(index=False).encode('utf-8')

    # Botón para descargar las predicciones
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )

    # Botón para resetear las predicciones
    if st.button("Reset Predictions"):
        st.session_state['predictions'] = []
        st.write("Predictions have been reset.")

    # Botón para descargar el informe en PDF
    with open(ml_report_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(
        label="📥 Download ML Model Report",
        data=PDFbyte,
        file_name="ML_Model_Report.pdf",
        mime='application/pdf',
    )

elif choice == "Deep Learning":
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
        cols[i % 2].image(image, caption=f"Label: {dl_class_names[row['label']]}", use_container_width=True)

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

    # Botón para descargar el informe en PDF
    with open(dl_report_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(
        label="📥 Download DL Model Report",
        data=PDFbyte,
        file_name="DL_Model_Report.pdf",
        mime='application/pdf',
    )

elif choice == "Chat Bot":
    st.header("Chat Bot")
    st.markdown("""
    This section will feature a chat bot to assist users with questions and provide support related to Alzheimer's disease.
    """)

    # Entrada de texto para el usuario
    user_input = st.text_input("You: ", "")

    # Inicializar bot_response
    bot_response = ""

    if user_input:
        # Generar el prompt específico
        prompt = generate_prompt(user_input)
        
        # Obtener la respuesta del chat bot
        bot_response = get_cohere_response(prompt)
        
        # Mostrar la conversación en un formato más sencillo
        st.write(f"**You:** {user_input}")
        st.write(f"**Chat Bot:** {bot_response}")

elif choice == "Other Resources":
    st.header("Other Resources")
    st.markdown("""
    Here you can find additional resources and information related to Alzheimer's disease, including research papers, support groups, and more.
    """)

     # Definir las rutas de los archivos PDF
    nutrition_guidelines_path = ('App/Nutrition_Guidelines.pdf')
    smoking_alzheimer_path = ('App/Smoking_and_Alzheimer.pdf')
    lysine_alzheimer_path = ('App/Lysine_and_Alzheimer.pdf')

    # Introducción y botón para descargar el PDF de pautas de alimentación
    st.subheader("🥗 Nutrition Guidelines")
    st.markdown("""
    This document provides guidelines on nutrition and dietary habits that may help in managing and potentially reducing the risk of Alzheimer's disease.
    """)
    with open(nutrition_guidelines_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    st.download_button(
        label="📥 Download Nutrition Guidelines PDF",
        data=PDFbyte,
        file_name="Nutrition_Guidelines.pdf",
        mime='application/pdf',
    )

    # Introducción y botón para descargar el PDF de la relación entre Alzheimer y fumar
    st.subheader("🚬 Smoking and Alzheimer's Disease")
    st.markdown("""
    This document explores the relationship between smoking and Alzheimer's disease, discussing how smoking may impact the risk and progression of the disease.
    """)
    with open(smoking_alzheimer_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    st.download_button(
        label="📥 Download Smoking and Alzheimer PDF",
        data=PDFbyte,
        file_name="Smoking_and_Alzheimer.pdf",
        mime='application/pdf',
    )

    # Introducción y botón para descargar el PDF de la relación entre Alzheimer y lisina
    st.subheader("💊 Lysine and Alzheimer's Disease")
    st.markdown("""
    This document examines the potential relationship between lysine, an essential amino acid, and Alzheimer's disease, including its possible effects on brain health.
    """)
    with open(lysine_alzheimer_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    st.download_button(
        label="📥 Download Lysine and Alzheimer PDF",
        data=PDFbyte,
        file_name="Lysine_and_Alzheimer.pdf",
        mime='application/pdf',
    )
    

# Footer
st.markdown("""
---
**Note:** This application is for educational purposes only and should not be used for real medical diagnoses.
""")