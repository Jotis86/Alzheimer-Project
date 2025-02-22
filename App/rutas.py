import os

# Obtener la ruta base relativa al script actual
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Sube un nivel

# Definir las rutas de los archivos dentro de la carpeta "App"
model_path = os.path.join(base_dir, 'App', 'alzheimer_model.h5')
train_data_file_path = os.path.join(base_dir, 'App', 'train.parquet')
test_data_file_path = os.path.join(base_dir, 'App', 'test.parquet')
ml_model_path = os.path.join(base_dir, 'App', 'best_random_forest_model.pkl')
feature_selector_path = os.path.join(base_dir, 'App', 'feature_selector.pkl')
navigation_image_path = os.path.join(base_dir, 'App', 'image_3.jpeg')
home_image_path = os.path.join(base_dir, 'App', 'image_2.jpeg')
ml_report_path = os.path.join(base_dir, 'App', 'ML_Report.pdf')
dl_report_path = os.path.join(base_dir, 'App', 'DL_Report.pdf')

# Verificar si las rutas son correctas
print("Model Path:", model_path)
print("Train Data Path:", train_data_file_path)
print("Test Data Path:", test_data_file_path)
print("ML Model Path:", ml_model_path)
print("Feature Selector Path:", feature_selector_path)
print("Navigation Image Path:", navigation_image_path)
print("Home Image Path:", home_image_path)
print("ML Report Path:", ml_report_path)
print("DL Report Path:", dl_report_path)