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


# Configuración de la página
st.set_page_config(
    page_title="Alzheimer AI: Detect & Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Jotis86/Alzheimer-Project/issues',
        'Report a bug': 'https://github.com/Jotis86/Alzheimer-Project/issues',
        'About': "# Alzheimer AI: Detect & Support\nThis app provides tools for early detection and management of Alzheimer's disease."
    }
)

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




# Título de la aplicación con banner de gradiente verde y sidebar con el mismo gradiente
st.markdown(
    """
    <style>
    .banner {
        background: linear-gradient(90deg, #006064, #1b5e20, #a5d6a7);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        font-style: italic;
        opacity: 0.9;
    }
    /* Estilo para la barra lateral con el mismo gradiente verde */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #006064, #1b5e20, #a5d6a7);
        color: white;
    }
    /* Estilo para el menú de navegación */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
    }
    
    .stRadio label {
        color: white !important;
        font-weight: 500;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 8px;
        display: flex;
        align-items: center;
        transition: all 0.3s;
    }
    
    .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Estilo para la sección del equipo */
    .team-section {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #333;
        margin-top: 20px;
    }
    
    .team-title {
        text-align: center;
        border-bottom: 2px solid rgba(255, 255, 255, 0.5);
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-weight: bold;
        color: #006064;
    }
    
    /* Estilos para el banner del sidebar */
    .sidebar-banner {
        background: linear-gradient(135deg, #006064, #1b5e20);
        border-radius: 10px;
        padding: 20px 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><path d="M80,20 Q100,40 80,60 Q60,80 40,60 Q20,40 40,20 Q60,0 80,20 Z" fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="2"/></svg>');
        background-size: 150px;
        opacity: 0.3;
    }
    
    .banner-logo {
        font-size: 32px;
        margin-bottom: 5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .banner-title {
        color: white;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .banner-subtitle {
        color: rgba(255, 255, 255, 0.85);
        font-size: 14px;
        font-style: italic;
    }
    
    .banner-dots {
        position: absolute;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 3px, transparent 4px);
        background-size: 30px 30px;
        transform: rotate(45deg);
        top: -75px;
        right: -75px;
        opacity: 0.4;
    }
    
    /* Estilo para el footer */
    .sidebar-footer {
        background: linear-gradient(180deg, rgba(0, 96, 100, 0.1), rgba(0, 96, 100, 0.3));
        color: white;
        text-align: center;
        padding: 15px 10px;
        font-size: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        margin-top: 20px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .footer-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        margin-bottom: 5px;
    }
    
    .footer-links a {
        color: white;
        margin: 0 10px;
        text-decoration: none;
        transition: all 0.3s;
    }
    
    .footer-links a:hover {
        color: #a5d6a7;
        transform: translateY(-2px);
    }
    
    .footer-text {
        opacity: 0.8;
    }
    
    .footer-divider {
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
        margin: 5px auto;
        border-radius: 2px;
    }
    </style>
    
    <div class="banner">
        <div class="title">🧠 Alzheimer AI: Detect & Support 🧠</div>
        <div class="subtitle">Advanced AI tools for early detection and management</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Banner personalizado para el sidebar
st.sidebar.markdown("""
<div class="sidebar-banner">
    <div class="banner-dots"></div>
    <div class="banner-logo">🧠</div>
    <div class="banner-title">Alzheimer AI</div>
    <div class="banner-subtitle">Detection & Support System</div>
</div>
""", unsafe_allow_html=True)

# Menú simplificado con opciones e iconos
menu_options = [
    "🏠 Home",
    "📊 Power BI",
    "🤖 Machine Learning", 
    "🧠 Deep Learning",
    "💬 Chat Bot",
    "📚 Other Resources"
]

# Usar el radio button estándar de Streamlit
choice = st.sidebar.radio("Navigate", menu_options, label_visibility="collapsed")

# Tabla con los nombres de los miembros del proyecto
st.sidebar.markdown('<div class="team-section">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="team-title">Project Members</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
| Name | Role |
|------|------|
| [Juan Duran](https://github.com/Jotis86) | Data Analyst 📊 |
| [Andrea Lafarga](https://github.com/AndreaLaHe) | Data Engineer 🛠️ |
""", unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Footer en el sidebar
st.sidebar.markdown("""
<div class="sidebar-footer">
    <div class="footer-content">
        <div class="footer-links">
            <a href="https://github.com/Jotis86/Alzheimer-Project" target="_blank">📂 GitHub</a>
            <a href="mailto:jotaduranbon@gmail.com">📧 Contact</a>
        </div>
        <div class="footer-divider"></div>
        <div class="footer-text">
            © 2023 Alzheimer AI Project • Made with ❤️ by Data lovers
        </div>
    </div>
</div>
""", unsafe_allow_html=True)




if choice == "🏠 Home":
    # Estilo para las tarjetas y elementos visuales
    st.markdown("""
    <style>
    /* Estilo para las tarjetas */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 2rem;
    }
    .info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 4px solid #1b5e20;
        margin-bottom: 16px;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #006064;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .card-title-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .card-content {
        color: #333;
        font-size: 1rem;
        line-height: 1.6;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 15px;
        color: #1b5e20;
        min-width: 30px;
        text-align: center;
    }
    .feature-text {
        flex-grow: 1;
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #2e7d32;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Separador decorativo */
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #006064, #1b5e20, #a5d6a7);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Estilo para la sección de llamada a la acción */
    .cta-card {
        margin-top: 16px; 
        text-align: center; 
        background: linear-gradient(90deg, rgba(0,96,100,0.1) 0%, rgba(27,94,32,0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .cta-title {
        font-size: 1.2rem; 
        margin-bottom: 10px;
        color: #006064;
        font-weight: bold;
    }
    .cta-text {
        color: #006064; 
        margin-bottom: 15px;
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Título principal
    #st.header("Welcome to Alzheimer AI: Detection and Support")
    
    # Sección de objetivos
    st.markdown('<div class="card-title"><span class="card-title-icon">🎯</span> Objectives of the Project</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🕵️‍♂️</div>
                <div class="feature-text">
                    <div class="feature-title">Early Detection</div>
                    <div class="feature-description">Utilize state-of-the-art models to detect Alzheimer's disease at an early stage.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🌐</div>
                <div class="feature-text">
                    <div class="feature-title">Accessibility</div>
                    <div class="feature-description">Provide an easy-to-use interface for healthcare professionals and researchers.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">📚</div>
                <div class="feature-text">
                    <div class="feature-title">Education</div>
                    <div class="feature-description">Raise awareness and educate users about Alzheimer's disease and its early signs.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Sección de características
    st.markdown('<div class="card-title"><span class="card-title-icon">✨</span> Features of This Application</div>', unsafe_allow_html=True)
    
    # Características con cada una en su propia tarjeta
    col1, col2 = st.columns(2)
    
    with col1:
        # Tarjeta 1: Power BI
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">📈</div>
                <div class="feature-text">
                    <div class="feature-title">Power BI Dashboards</div>
                    <div class="feature-description">Visualize data and gain insights through interactive dashboards that help identify patterns and trends.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tarjeta 2: Machine Learning
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-text">
                    <div class="feature-title">Machine Learning Model</div>
                    <div class="feature-description">Input various clinical and demographic features to get a prediction based on advanced algorithms.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Tarjeta 3: Deep Learning
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🖼️</div>
                <div class="feature-text">
                    <div class="feature-title">Deep Learning Model</div>
                    <div class="feature-description">Upload MRI scans for automated analysis and detection of Alzheimer's patterns in brain images.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tarjeta 4: Chat Bot
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">💬</div>
                <div class="feature-text">
                    <div class="feature-title">Chat Bot</div>
                    <div class="feature-description">Get answers to your questions and support related to Alzheimer's disease from our AI assistant.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sección final con llamada a la acción - Con mejor legibilidad
    st.markdown("""
    <div class="cta-card">
        <div class="cta-title">
            <span style="font-size: 1.5rem;">🚀</span> Ready to explore?
        </div>
        <div class="cta-text">
            Use the navigation menu on the left to discover all the tools and resources available for Alzheimer's detection and support.
        </div>
    </div>
    """, unsafe_allow_html=True)




elif choice == "📊 Power BI":
    # Usamos exactamente los mismos estilos que en Home para mantener coherencia
    st.markdown("""
    <style>
    /* Estilo para las tarjetas */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 2rem;
    }
    .info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 4px solid #1b5e20;
        margin-bottom: 16px;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #006064;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .card-title-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .card-content {
        color: #333;
        font-size: 1rem;
        line-height: 1.6;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 15px;
        color: #1b5e20;
        min-width: 30px;
        text-align: center;
    }
    .feature-text {
        flex-grow: 1;
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #2e7d32;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Separador decorativo */
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #006064, #1b5e20, #a5d6a7);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Estilo para la sección de llamada a la acción */
    .cta-card {
        margin-top: 16px; 
        text-align: center; 
        background: linear-gradient(90deg, rgba(0,96,100,0.1) 0%, rgba(27,94,32,0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .cta-title {
        font-size: 1.2rem; 
        margin-bottom: 10px;
        color: #006064;
        font-weight: bold;
    }
    .cta-text {
        color: #006064; 
        margin-bottom: 15px;
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Título principal
    #st.header("Power BI Dashboards")
    
    # Introducción con el mismo estilo de tarjeta info-card
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-text">
                <div class="feature-title">Interactive Data Visualizations</div>
                <div class="feature-description">
                    Explore our comprehensive Power BI dashboards that provide deep insights into Alzheimer's disease data.
                    These interactive visualizations help healthcare professionals and researchers better understand patterns,
                    trends, and correlations in the data.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de categorías de dashboards
    st.markdown('<div class="card-title"><span class="card-title-icon">📊</span> Dashboard Categories</div>', unsafe_allow_html=True)
    
    # Tarjetas para cada categoría - 3 columnas como en el Home
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Tarjeta 1: Lifestyle Analysis
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🍎</div>
                <div class="feature-text">
                    <div class="feature-title">Lifestyle Analysis</div>
                    <div class="feature-description">Explore the impact of lifestyle factors on Alzheimer's disease risk and progression.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Tarjeta 2: Medical Analysis
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🩺</div>
                <div class="feature-text">
                    <div class="feature-title">Medical Analysis</div>
                    <div class="feature-description">Visualize clinical measurements and their relationship to Alzheimer's detection.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Tarjeta 3: Cognitive Analysis
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🧠</div>
                <div class="feature-text">
                    <div class="feature-title">Cognitive Analysis</div>
                    <div class="feature-description">Examine cognitive assessments to identify early warning signs of Alzheimer's.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Sección de características detalladas
    st.markdown('<div class="card-title"><span class="card-title-icon">🔍</span> Dashboard Features</div>', unsafe_allow_html=True)
    
    # Características detalladas - 2 columnas como en Home
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">📈</div>
                <div class="feature-text">
                    <div class="feature-title">Interactive Filters</div>
                    <div class="feature-description">
                        <p>Our dashboards include interactive filters:</p>
                        <ul style="margin-top: 10px; padding-left: 20px;">
                            <li>Filter by age, gender, and other demographics</li>
                            <li>Slice data by different cognitive assessment scores</li>
                            <li>Compare groups with similar characteristics</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🔄</div>
                <div class="feature-text">
                    <div class="feature-title">Data Exploration</div>
                    <div class="feature-description">
                        <p>Powerful exploration capabilities:</p>
                        <ul style="margin-top: 10px; padding-left: 20px;">
                            <li>Drill down into specific data points efficiently</li>
                            <li>Explore trends over time seamlessly and intuitively</li>
                            <li>Export visualizations and insights instantly</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Sección de video
    st.markdown('<div class="card-title"><span class="card-title-icon">🎥</span> Dashboard Preview</div>', unsafe_allow_html=True)
    
    # Video como una tarjeta única que ocupa toda la anchura
    st.markdown("""
    <div class="info-card">
        <p style="color: #444; margin-bottom: 15px;">
            Watch this short video demonstration of our Power BI dashboards to see their capabilities in action:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reproducir el video de Power BI
    video = os.path.join('Power_BI', '2025-03-02 19-23-46.mp4')
    st.video(video)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Sección de descarga
    st.markdown('<div class="card-title"><span class="card-title-icon">💾</span> Download Resources</div>', unsafe_allow_html=True)
    
    # Columnas para mejor disposición
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Información de descarga
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">📥</div>
                <div class="feature-text">
                    <div class="feature-title">Power BI Dashboard Files</div>
                    <div class="feature-description">
                        Download the complete Power BI dashboard file to explore the data on your own computer. 
                        You can open these files with Power BI Desktop, which is free and available for Windows.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Tarjeta para el botón de descarga
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="font-weight: 600; color: #2e7d32; margin-bottom: 15px;">
                Download Dashboard
            </div>
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 15px;">
                Click the button below to get the full Power BI file
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para descargar el archivo de Power BI
        pbix_path = os.path.join('Power_BI', 'Alzheimer_Dashboard.pbix')
        with open(pbix_path, "rb") as pbix_file:
            PBIXbyte = pbix_file.read()
        
        st.download_button(
            label="📥 Download Dashboard",
            data=PBIXbyte,
            file_name="Alzheimer_Dashboard.pbix",
            mime='application/vnd.powerbi.pbix',
            help="Click to download the complete Power BI dashboard file"
        )
    
    # Nota final con estilo CTA como en Home
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">ℹ️</div>
            <div class="feature-text">
                <div class="feature-title">Important Note</div>
                <div class="feature-description">
                    To open these files, you'll need <a href="https://powerbi.microsoft.com/desktop/" target="_blank">Power BI Desktop</a> 
                    installed on your computer. The application is free and available for Windows.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)





elif choice == "🤖 Machine Learning":
    # Definición de estilos igual que en las otras secciones
    st.markdown("""
    <style>
    /* Estilo para las tarjetas */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 2rem;
    }
    .info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 4px solid #1b5e20;
        margin-bottom: 16px;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #006064;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .card-title-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .card-content {
        color: #333;
        font-size: 1rem;
        line-height: 1.6;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 15px;
        color: #1b5e20;
        min-width: 30px;
        text-align: center;
    }
    .feature-text {
        flex-grow: 1;
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #2e7d32;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Separador decorativo */
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #006064, #1b5e20, #a5d6a7);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Estilo para la sección de llamada a la acción */
    .cta-card {
        margin-top: 16px; 
        text-align: center; 
        background: linear-gradient(90deg, rgba(0,96,100,0.1) 0%, rgba(27,94,32,0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .cta-title {
        font-size: 1.2rem; 
        margin-bottom: 10px;
        color: #006064;
        font-weight: bold;
    }
    .cta-text {
        color: #006064; 
        margin-bottom: 15px;
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)

    #st.header("Machine Learning Model for Alzheimer's Detection")
    
    # Introducción con tarjeta estilo info-card
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">🧠</div>
            <div class="feature-text">
                <div class="feature-title">About the Model</div>
                <div class="feature-description">
                    This application uses a machine learning model trained on data from 2,149 patients to detect 
                    Alzheimer's disease. The dataset includes demographic details, lifestyle factors, medical history, 
                    clinical measurements, and cognitive assessments. Input your values below to get a prediction.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Título de la sección de variables
    st.markdown('<div class="card-title"><span class="card-title-icon">📋</span> Key Variables for Prediction</div>', unsafe_allow_html=True)
    
    # Variables Cognitivas - 3 columnas como en Home
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🧩</div>
                <div class="feature-text">
                    <div class="feature-title">MMSE Score</div>
                    <div class="feature-description">Mini-Mental State Examination score (0–30). Lower scores indicate cognitive impairment severity.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🔄</div>
                <div class="feature-text">
                    <div class="feature-title">ADL Score</div>
                    <div class="feature-description">Activities of Daily Living score (0-10). Measures ability to perform everyday tasks independently.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">📝</div>
                <div class="feature-text">
                    <div class="feature-title">Functional Assessment</div>
                    <div class="feature-description">Score ranging from 0 to 10. Lower scores indicate greater impairment in daily functions.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador pequeño
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Segunda fila de variables - 3 columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">❤️</div>
                <div class="feature-text">
                    <div class="feature-title">Blood Pressure</div>
                    <div class="feature-description">Systolic (90-180 mmHg) and Diastolic (60-120 mmHg) blood pressure measurements.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🩸</div>
                <div class="feature-text">
                    <div class="feature-title">HDL Cholesterol</div>
                    <div class="feature-description">HDL "good" cholesterol levels (20-100 mg/dL). Higher levels are generally better.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">💤</div>
                <div class="feature-text">
                    <div class="feature-title">Sleep Quality</div>
                    <div class="feature-description">Sleep quality score (1-10). Higher scores indicate better sleep quality and patterns.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador pequeño
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Tercera fila de variables - 3 columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🎓</div>
                <div class="feature-text">
                    <div class="feature-title">Education Level</div>
                    <div class="feature-description">0: None, 1: High School, 2: Bachelor's, 3: Higher education (Master's/PhD).</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🧠</div>
                <div class="feature-text">
                    <div class="feature-title">Memory Complaints</div>
                    <div class="feature-description">Presence of memory complaints reported by patient or caregiver (0: No, 1: Yes).</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">😔</div>
                <div class="feature-text">
                    <div class="feature-title">Behavioral Problems</div>
                    <div class="feature-description">Presence of behavioral issues like agitation, aggression, or mood changes (0: No, 1: Yes).</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Sección de predicción
    st.markdown('<div class="card-title"><span class="card-title-icon">🔮</span> Make a Prediction</div>', unsafe_allow_html=True)
    
    # Introducción para la sección de predicción
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">ℹ️</div>
            <div class="feature-text">
                <div class="feature-title">How to Use</div>
                <div class="feature-description">
                    Enter the values for each variable in the form below, then click "Predict" to get the model's assessment. 
                    This tool is for informational purposes only and should not replace professional medical advice.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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





elif choice == "🧠 Deep Learning":
    # Definición de estilos igual que en las otras secciones
    st.markdown("""
    <style>
    /* Estilo para las tarjetas */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 2rem;
    }
    .info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 4px solid #1b5e20;
        margin-bottom: 16px;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #006064;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .card-title-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .card-content {
        color: #333;
        font-size: 1rem;
        line-height: 1.6;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 15px;
        color: #1b5e20;
        min-width: 30px;
        text-align: center;
    }
    .feature-text {
        flex-grow: 1;
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #2e7d32;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Separador decorativo */
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #006064, #1b5e20, #a5d6a7);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Estilo para la sección de llamada a la acción */
    .cta-card {
        margin-top: 16px; 
        text-align: center; 
        background: linear-gradient(90deg, rgba(0,96,100,0.1) 0%, rgba(27,94,32,0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .cta-title {
        font-size: 1.2rem; 
        margin-bottom: 10px;
        color: #006064;
        font-weight: bold;
    }
    .cta-text {
        color: #006064; 
        margin-bottom: 15px;
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("Deep Learning Model for Alzheimer's Detection")
    
    # Introducción con tarjeta estilo info-card
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">🖼️</div>
            <div class="feature-text">
                <div class="feature-title">MRI Image Analysis</div>
                <div class="feature-description">
                    This application uses a deep learning model to detect Alzheimer's disease from MRI brain scan images.
                    The model analyzes patterns in brain structure that may indicate different stages of dementia.
                    You can select an image from our test dataset, and the AI will classify it into one of four categories.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Título de la sección de categorías
    st.markdown('<div class="card-title"><span class="card-title-icon">🔍</span> Classification Categories</div>', unsafe_allow_html=True)
    
    # Tarjetas para cada categoría de clasificación - 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🟢</div>
                <div class="feature-text">
                    <div class="feature-title">Non-Demented</div>
                    <div class="feature-description">
                        MRI scans that show normal brain structure without significant signs of neurodegeneration 
                        associated with Alzheimer's disease. These images serve as the baseline for comparison.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🟠</div>
                <div class="feature-text">
                    <div class="feature-title">Moderate Demented</div>
                    <div class="feature-description">
                        Images showing advanced neurodegeneration with significant brain atrophy, enlarged ventricles, and cortical thinning. Moderate dementia presents with substantial memory and cognitive deficits.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🟡</div>
                <div class="feature-text">
                    <div class="feature-title">Very Mild Demented</div>
                    <div class="feature-description">
                        Brain scans showing very early signs of neurodegeneration that may be difficult to detect visually. 
                        These subtle changes may include minor hippocampal atrophy and early ventricular enlargement.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🔴</div>
                <div class="feature-text">
                    <div class="feature-title">Mild Demented</div>
                    <div class="feature-description">
                        MRI scans showing early but definite signs of Alzheimer's-related neurodegeneration, including 
                        hippocampal atrophy, widened sulci, and some ventricular enlargement typical of early-stage Alzheimer's.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Título de la sección de dataset
    st.markdown('<div class="card-title"><span class="card-title-icon">📊</span> Dataset Overview</div>', unsafe_allow_html=True)
    
    # Información del dataset en tarjetas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">📈</div>
                <div class="feature-text">
                    <div class="feature-title">Dataset Statistics</div>
                    <div class="feature-description">
                        <ul style="padding-left: 20px; margin-top: 5px;">
                            <li><strong>Total Images:</strong> 6,400</li>
                            <li><strong>Training Set:</strong> 5,120 images (80%)</li>
                            <li><strong>Test Set:</strong> 1,280 images (20%)</li>
                            <li><strong>Categories:</strong> 4 classes of dementia</li>
                            <li><strong>Image Type:</strong> Brain MRI scans</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🤖</div>
                <div class="feature-text">
                    <div class="feature-title">Model Architecture</div>
                    <div class="feature-description">
                        <p>Our deep learning model uses a convolutional neural network (CNN) architecture specifically optimized for medical image analysis:</p>
                        <ul style="padding-left: 20px; margin-top: 5px;">
                            <li>Pre-trained on medical imaging datasets</li>
                            <li>Fine-tuned on Alzheimer's MRI dataset</li>
                            <li>Achieves >95% accuracy on the test set</li>
                            <li>Focuses on brain regions most affected by Alzheimer's</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Sección de instrucciones de uso
    st.markdown('<div class="card-title"><span class="card-title-icon">📝</span> How to Use the Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">ℹ️</div>
            <div class="feature-text">
                <div class="feature-title">Instructions</div>
                <div class="feature-description">
                    <ol style="padding-left: 20px; margin-top: 5px;">
                        <li>Select an MRI scan from our test dataset using the selector below</li>
                        <li>The image will be preprocessed automatically</li>
                        <li>Click "Analyze Image" to run the deep learning model</li>
                        <li>Review the results showing predicted class and confidence score</li>
                        <li>Note: This tool is for research and educational purposes only</li>
                    </ol>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

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

elif choice == "💬 Chat Bot":
    # Definición de estilos igual que en las otras secciones
    st.markdown("""
    <style>
    /* Estilo para las tarjetas */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 2rem;
    }
    .info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 4px solid #1b5e20;
        margin-bottom: 16px;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #006064;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .card-title-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .card-content {
        color: #333;
        font-size: 1rem;
        line-height: 1.6;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 15px;
        color: #1b5e20;
        min-width: 30px;
        text-align: center;
    }
    .feature-text {
        flex-grow: 1;
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #2e7d32;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Separador decorativo */
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #006064, #1b5e20, #a5d6a7);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Estilo para la sección de llamada a la acción */
    .cta-card {
        margin-top: 16px; 
        text-align: center; 
        background: linear-gradient(90deg, rgba(0,96,100,0.1) 0%, rgba(27,94,32,0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .cta-title {
        font-size: 1.2rem; 
        margin-bottom: 10px;
        color: #006064;
        font-weight: bold;
    }
    .cta-text {
        color: #006064; 
        margin-bottom: 15px;
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)

    #st.header("Alzheimer's AI Assistant")
    
    # Introducción con tarjeta estilo info-card
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">🤖</div>
            <div class="feature-text">
                <div class="feature-title">Your AI Support Companion</div>
                <div class="feature-description">
                    Our AI Assistant is designed to provide information, answer questions, and offer support related to Alzheimer's disease.
                    Whether you're a patient, caregiver, family member, or healthcare professional, this chatbot can help you find 
                    the information you need in a conversational format.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

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

elif choice == "📚 Other Resources":
    # Definición de estilos igual que en las otras secciones
    st.markdown("""
    <style>
    /* Estilo para las tarjetas */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 2rem;
    }
    .info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 4px solid #1b5e20;
        margin-bottom: 16px;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #006064;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .card-title-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .card-content {
        color: #333;
        font-size: 1rem;
        line-height: 1.6;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 15px;
        color: #1b5e20;
        min-width: 30px;
        text-align: center;
    }
    .feature-text {
        flex-grow: 1;
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #2e7d32;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Separador decorativo */
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #006064, #1b5e20, #a5d6a7);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Estilo para la sección de llamada a la acción */
    .cta-card {
        margin-top: 16px; 
        text-align: center; 
        background: linear-gradient(90deg, rgba(0,96,100,0.1) 0%, rgba(27,94,32,0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .cta-title {
        font-size: 1.2rem; 
        margin-bottom: 10px;
        color: #006064;
        font-weight: bold;
    }
    .cta-text {
        color: #006064; 
        margin-bottom: 15px;
        font-size: 1.05rem;
    }
    
    /* Estilo para botones de recursos */
    .resource-button {
        text-align: center;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("Additional Resources for Alzheimer's")
    
    # Introducción con tarjeta estilo info-card
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">📚</div>
            <div class="feature-text">
                <div class="feature-title">Research & Educational Materials</div>
                <div class="feature-description">
                    Explore our collection of resources related to Alzheimer's disease. These materials provide valuable insights 
                    into lifestyle factors, dietary considerations, and other aspects that may affect brain health and Alzheimer's risk.
                    All documents are available for download in PDF format.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Título de la sección de documentos
    st.markdown('<div class="card-title"><span class="card-title-icon">📄</span> Research Documents</div>', unsafe_allow_html=True)
    
    # Definir las rutas de los archivos PDF
    nutrition_guidelines_path = ('App/Nutrition_Guidelines.pdf')
    smoking_alzheimer_path = ('App/Smoking_and_Alzheimer.pdf')
    lysine_alzheimer_path = ('App/Lysine_and_Alzheimer.pdf')

    # Organizar los recursos en 3 columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Recurso 1: Nutrition Guidelines
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🥗</div>
                <div class="feature-text">
                    <div class="feature-title">Nutrition Guidelines</div>
                    <div class="feature-description">
                        Comprehensive guidelines on brain-healthy nutrition that may help reduce Alzheimer's risk. 
                        Includes recommended food groups, meal planning suggestions, and scientific rationale.
                    </div>
                </div>
            </div>
            <div class="resource-button">
                <img src="https://img.icons8.com/color/48/000000/pdf.png" width="32" style="vertical-align: middle; margin-right: 5px;">
                <span style="color: #1b5e20; font-weight: 500;">Nutrition Guidelines PDF</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with open(nutrition_guidelines_path, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button(
            label="📥 Download PDF",
            data=PDFbyte,
            file_name="Nutrition_Guidelines.pdf",
            mime='application/pdf',
            key="nutrition_pdf"
        )
    
    with col2:
        # Recurso 2: Smoking and Alzheimer's
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🚬</div>
                <div class="feature-text">
                    <div class="feature-title">Smoking risk</div>
                    <div class="feature-description">
                        Research summary on the relationship between smoking and Alzheimer's disease risk. 
                        Examines the mechanisms by which tobacco use may accelerate cognitive decline and neurodegeneration.
                    </div>
                </div>
            </div>
            <div class="resource-button">
                <img src="https://img.icons8.com/color/48/000000/pdf.png" width="32" style="vertical-align: middle; margin-right: 5px;">
                <span style="color: #1b5e20; font-weight: 500;">Smoking & Alzheimer's PDF</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with open(smoking_alzheimer_path, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button(
            label="📥 Download PDF",
            data=PDFbyte,
            file_name="Smoking_and_Alzheimer.pdf",
            mime='application/pdf',
            key="smoking_pdf"
        )
    
    with col3:
        # Recurso 3: Lysine and Alzheimer's
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">💊</div>
                <div class="feature-text">
                    <div class="feature-title">Lysine & Alzheimer's</div>
                    <div class="feature-description">
                        Scientific analysis of lysine's potential role in brain health and Alzheimer's disease. 
                        Covers recent research on this essential amino acid and its effects on neurological function.
                    </div>
                </div>
            </div>
            <div class="resource-button">
                <img src="https://img.icons8.com/color/48/000000/pdf.png" width="32" style="vertical-align: middle; margin-right: 5px;">
                <span style="color: #1b5e20; font-weight: 500;">Lysine & Alzheimer's PDF</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with open(lysine_alzheimer_path, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button(
            label="📥 Download PDF",
            data=PDFbyte,
            file_name="Lysine_and_Alzheimer.pdf",
            mime='application/pdf',
            key="lysine_pdf"
        )
    
    # Separador decorativo
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Título de la sección de enlaces
    st.markdown('<div class="card-title"><span class="card-title-icon">🔗</span> Useful Links & Support Groups</div>', unsafe_allow_html=True)
    
    # Enlaces útiles en 2 columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">🏥</div>
                <div class="feature-text">
                    <div class="feature-title">Medical Resources</div>
                    <div class="feature-description">
                        <ul style="padding-left: 20px; margin-top: 5px;">
                            <li><a href="https://www.alz.org/" target="_blank">Alzheimer's Association</a> - Research, care and support</li>
                            <li><a href="https://www.nia.nih.gov/health/alzheimers" target="_blank">National Institute on Aging</a> - Government resources</li>
                            <li><a href="https://www.who.int/news-room/fact-sheets/detail/dementia" target="_blank">World Health Organization</a> - Global health perspective</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-item">
                <div class="feature-icon">👨‍👩‍👧‍👦</div>
                <div class="feature-text">
                    <div class="feature-title">Support Groups</div>
                    <div class="feature-description">
                        <ul style="padding-left: 20px; margin-top: 5px;">
                            <li><a href="https://www.alz.org/help-support/community/support-groups" target="_blank">Alzheimer's Association Support Groups</a></li>
                            <li><a href="https://www.alzconnected.org/" target="_blank">AlzConnected</a> - Online community</li>
                            <li><a href="https://www.caregiver.org/connecting-caregivers/support-groups/" target="_blank">Family Caregiver Alliance</a> - Caregiver support</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Nota final
    st.markdown("""
    <div class="info-card">
        <div class="feature-item">
            <div class="feature-icon">📝</div>
            <div class="feature-text">
                <div class="feature-title">Request Additional Resources</div>
                <div class="feature-description">
                    If you're looking for specific information not found here, please contact us. We regularly update our resource 
                    library based on new research and user requests. All materials are carefully reviewed by our team of medical professionals 
                    to ensure accuracy and relevance.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

# Footer
st.markdown("""
---
**Note:** This application is for educational purposes only and should not be used for real medical diagnoses.
""")