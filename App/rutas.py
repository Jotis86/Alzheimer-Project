import os

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Imprimir el directorio actual
print(f"El directorio actual es: {current_dir}")

# Rutas de los archivos
model_path = os.path.join(current_dir, 'alzheimer_model.h5')
train_data_file_path = os.path.join(current_dir, 'train.parquet')
test_data_file_path = os.path.join(current_dir, 'test.parquet')
ml_model_path = os.path.join(current_dir, 'best_random_forest_model.pkl')
feature_selector_path = os.path.join(current_dir, 'feature_selector.pkl')
navigation_image_path = os.path.join(current_dir, 'image_3.jpeg')
home_image_path = os.path.join(current_dir, 'image_2.jpeg')
ml_report_path = os.path.join(current_dir, 'ML_Report.pdf')
dl_report_path = os.path.join(current_dir, 'DL_Report.pdf')

# Lista de rutas para comprobar
paths = [
    model_path,
    train_data_file_path,
    test_data_file_path,
    ml_model_path,
    feature_selector_path,
    navigation_image_path,
    home_image_path,
    ml_report_path,
    dl_report_path
]

# Comprobar si los archivos existen y mostrar la ruta
for path in paths:
    if os.path.exists(path):
        print(f"El archivo existe: {path}")
    else:
        print(f"Archivo NO encontrado: {path}")