import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt  # Importar Keras Tuner para optimizar hiperparámetros

#Definimos la ruta de la carpeta y la ruta del archivo csv
data_folder = 'D:/YOLO'
csv_file = os.path.join(data_folder,"train_samples.csv")

#Cargamos el archivo csv
df = pd.read_csv(csv_file)

# Filtramos las filas que correspondan a la condicion "Spinal Canal Stenosis"
df_filtered = df[df['condition'] == "Spinal Canal Stenosis"]

# Crear una nueva columna 'image_name' que mapea al nombre correcto de las imágenes en formato .jpg
df_filtered['image_name'] = df_filtered.apply(lambda row: f"{row['study_id']}_{row['series_id']}_{row['instance_number']}.jpg", axis=1)

# Definir la ruta de la carpeta de las imágenes
images_folder = os.path.join(data_folder, "train", "images")

# Definir las dimensiones a las que redimensionaremos las imágenes
img_height, img_width = 128, 128

# Lista para almacenar las imágenes y las etiquetas
images = []
labels = []

# Recorrer el dataframe filtrado y cargar las imágenes correspondientes
for index, row in df_filtered.iterrows():
    image_name = row['image_name']
    label = row['label']  # Usamos la columna 'label' que indica la severidad
    image_path = os.path.join(images_folder, image_name)

    try:
        img = Image.open(image_path)
        img = img.resize((img_width, img_height))  # Redimensionar la imagen
        img_array = img_to_array(img)  # Convertir la imagen a array de numpy
        img_array = img_array / 255.0  # Normalizar los píxeles (0-255 a 0-1)
        images.append(img_array)
        labels.append(label)
    except Exception as e:
        print(f"Error al cargar la imagen {image_name}: {e}")

# Convertir las listas en arrays de numpy
images = np.array(images)
labels = np.array(labels)

# Dividir los datos en entrenamiento y validación (70% entrenamiento, 30% validación)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, random_state=42)

# Función para crear el modelo con hiperparámetros ajustables
def build_model(hp):
    model = Sequential()

    # Primera capa convolucional con filtros ajustables
    model.add(Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(img_height, img_width, 1)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda capa convolucional
    model.add(Conv2D(
        filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Tercera capa convolucional
    model.add(Conv2D(
        filters=hp.Int('filters_3', min_value=64, max_value=256, step=64),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Aplanar y añadir capas densas
    model.add(Flatten())

    # Capa densa ajustable
    model.add(Dense(
        units=hp.Int('units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))

    # Añadir capa Dropout ajustable
    model.add(Dropout(
        rate=hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1)
    ))

    # Capa de salida
    model.add(Dense(4, activation='softmax'))

    # Compilar el modelo con tasa de aprendizaje ajustable
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Definir el tuner para buscar los mejores hiperparámetros
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Número de combinaciones a probar
    executions_per_trial=2,  # Ejecutar dos veces cada configuración
    directory='tuner_dir',
    project_name='lumbar_stenosis'
)

# Iniciar la búsqueda de los mejores hiperparámetros
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Ver los mejores hiperparámetros encontrados
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Mejores hiperparámetros: {best_hps.values}")

# Entrenar el mejor modelo encontrado
best_model = tuner.hypermodel.build(best_hps)

# Añadir early stopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenar el modelo final con los mejores hiperparámetros
history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
                         batch_size=32, callbacks=[early_stopping])

# Resumen del modelo
best_model.summary()
