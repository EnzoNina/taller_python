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



#Definimos la ruta de la ccarpeta y la ruta del archivo csv
data_folder = 'D:/YOLO'
csv_file = os.path.join(data_folder,"train_samples.csv")

#Cargamos el archivo csv
df = pd.read_csv(csv_file)

#Imprimimos información del dataframe
print(df.info())

#Filtramos las filas que correspondan a la condicion "Spinal Canal Stenosis"
df_filtered = df[df['condition'] == "Spinal Canal Stenosis"]

#Vemos cuantas filas tiene el dataframe filtrado
print(f"Total de imágenes filtradas para 'Spinal Canal Stenosis': {len(df_filtered)}")

# Crear una nueva columna 'image_name' que mapea al nombre correcto de las imágenes en formato .jpg
df_filtered['image_name'] = df_filtered.apply(lambda row: f"{row['study_id']}_{row['series_id']}_{row['instance_number']}.jpg", axis=1)

# Ver las primeras filas del dataframe con el mapeo correcto de las imágenes
print(df_filtered[['study_id', 'series_id', 'instance_number', 'image_name']].head())

#Definimos la ruta de la carpeta de las imágenes
images_folder = os.path.join(data_folder, "train", "images")

# Función para cargar y mostrar algunas imágenes filtradas
def load_image(image_name):
    full_path = os.path.join(images_folder, image_name)
    image = Image.open(full_path)
    return image

# Mostrar algunas imágenes para verificar
sample_images = df_filtered['image_name'].sample(5)  # Tomar 5 imágenes aleatorias para visualización

for img_name in sample_images:
    img = load_image(img_name)
    plt.imshow(img, cmap='gray')
    plt.title(f"Image: {img_name}")
    plt.axis('off')
    plt.show()


# Ruta de las imágenes
images_folder = "Dataset/train/images"

# Definir las dimensiones a las que redimensionaremos las imágenes
img_height, img_width = 128, 128

# Lista para almacenar las imágenes y las etiquetas
images = []
labels = []

# Recorrer el dataframe filtrado y cargar las imágenes correspondientes
for index, row in df_filtered.iterrows():
    # Crear el nombre del archivo de imagen
    image_name = row['image_name']
    label = row['label']  # Usamos la columna 'label' que indica la severidad

    # Ruta completa de la imagen
    image_path = os.path.join(images_folder, image_name)

    # Cargar la imagen, redimensionarla y convertirla a un array
    try:
        img = Image.open(image_path)
        img = img.resize((img_width, img_height))  # Redimensionar la imagen
        img_array = img_to_array(img)  # Convertir la imagen a array de numpy
        img_array = img_array / 255.0  # Normalizar los píxeles (0-255 a 0-1)

        # Añadir la imagen y la etiqueta a las listas
        images.append(img_array)
        labels.append(label)
    except Exception as e:
        print(f"Error al cargar la imagen {image_name}: {e}")

# Convertir las listas en arrays de numpy
images = np.array(images)
labels = np.array(labels)

# Dividir los datos en entrenamiento y validación (70% entrenamiento, 30% validación)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, random_state=42)

# Mostrar las dimensiones de los conjuntos
print(f"Conjunto de entrenamiento: {X_train.shape}")
print(f"Conjunto de validación: {X_val.shape}")

# Construir el modelo CNN
model = Sequential()

# Capa convolucional 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa convolucional 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa convolucional 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar y añadir capas densas
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))  # Regularización para evitar sobreajuste
model.add(Dense(4, activation='softmax'))  # 4 salidas (0: Normal, 1: Leve, 2: Moderado, 3: Severo)

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Definir el callback de Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenar el modelo con Early Stopping
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
                    batch_size=32, callbacks=[early_stopping])