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
data_folder = 'YOLO'
csv_file = os.path.join(data_folder, "train_samples.csv")

#Cargamos el archivo csv
df = pd.read_csv(csv_file)

# Filtramos las filas que correspondan a la condicion "Spinal Canal Stenosis"
df_filtered = df[df['condition'] == "Spinal Canal Stenosis"]

# Crear una nueva columna 'image_name' que mapea al nombre correcto de las imágenes en formato .jpg
df_filtered['image_name'] = df_filtered.apply(
    lambda row: f"{row['study_id']}_{row['series_id']}_{row['instance_number']}.jpg", axis=1)

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


# Función para contar imágenes por clase
def contar_imagenes_por_clase(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    class_names = ['Normal', 'Mild', 'Moderate', 'Severe']

    for label, count in class_counts.items():
        print(f"Clase {class_names[label]}: {count} imágenes")

    return class_counts


# Convertir las listas en arrays de numpy
images = np.array(images)
labels = np.array(labels)

# Contar imágenes por clase antes del aumento de datos
print("Imágenes por clase ANTES del aumento de datos:")
contar_imagenes_por_clase(labels)

# Filtrar las imágenes de las clases "Moderate" y "Severe"
moderate_images = images[labels == 2]  # 2 es la etiqueta para Moderate
moderate_labels = labels[labels == 2]

severe_images = images[labels == 3]  # 3 es la etiqueta para Severe
severe_labels = labels[labels == 3]

# Concatenar las imágenes y etiquetas de Moderate y Severe
minority_images = np.concatenate([moderate_images, severe_images], axis=0)
minority_labels = np.concatenate([moderate_labels, severe_labels], axis=0)

# Crear un generador de aumento de datos
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear un generador de aumento de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Definir cuántas imágenes queremos para cada clase
target_images_per_class = 9000

# Para la clase Moderate (etiqueta 2)
moderate_images = images[labels == 2]
moderate_labels = labels[labels == 2]
augmented_images_moderate = []
augmented_labels_moderate = []

# Generar más imágenes para Moderate
for x_batch, y_batch in datagen.flow(moderate_images, moderate_labels, batch_size=32, shuffle=False):
    augmented_images_moderate.append(x_batch)
    augmented_labels_moderate.append(y_batch)

    # Si llegamos a las 9000 imágenes (incluyendo las originales)
    if len(augmented_images_moderate) * 32 + len(moderate_images) >= target_images_per_class:
        break

# Convertir a arrays de numpy
augmented_images_moderate = np.vstack(augmented_images_moderate)
augmented_labels_moderate = np.full(augmented_images_moderate.shape[0], 2)

# Para la clase Severe (etiqueta 3)
severe_images = images[labels == 3]
severe_labels = labels[labels == 3]
augmented_images_severe = []
augmented_labels_severe = []

# Generar más imágenes para Severe
for x_batch, y_batch in datagen.flow(severe_images, severe_labels, batch_size=32, shuffle=False):
    augmented_images_severe.append(x_batch)
    augmented_labels_severe.append(y_batch)

    # Si llegamos a las 9000 imágenes (incluyendo las originales)
    if len(augmented_images_severe) * 32 + len(severe_images) >= target_images_per_class:
        break

# Convertir a arrays de numpy
augmented_images_severe = np.vstack(augmented_images_severe)
augmented_labels_severe = np.full(augmented_images_severe.shape[0], 3)

# Combinar los datos originales con los datos aumentados
images_augmented = np.concatenate([images, augmented_images_moderate, augmented_images_severe], axis=0)
labels_augmented = np.concatenate([labels, augmented_labels_moderate, augmented_labels_severe], axis=0)


# Verificar cuántas imágenes hay por clase después del aumento
def contar_imagenes_por_clase(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    class_names = ['Normal', 'Mild', 'Moderate', 'Severe']

    for label, count in class_counts.items():
        print(f"Clase {class_names[label]}: {count} imágenes")


# Contar imágenes por clase después del aumento de datos
print("Imágenes por clase DESPUÉS del aumento de datos:")
contar_imagenes_por_clase(labels_augmented)

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

# Evaluar el modelo en el conjunto de validación
loss, accuracy = best_model.evaluate(X_val, y_val)
print(f"Pérdida en validación: {loss}")
print(f"Precisión en validación: {accuracy}")

# Predicciones en el conjunto de validación
y_pred = np.argmax(best_model.predict(X_val), axis=1)

# Matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Mild', 'Moderate', 'Severe'],
            yticklabels=['Mild', 'Moderate', 'Severe'])
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión')
plt.show()

# Informe de clasificación
print(classification_report(y_val, y_pred, target_names=['Mild', 'Moderate', 'Severe']))

# Resumen del modelo
best_model.summary()

#Guardamos el modelo para usarlo en otro archivo
best_model.save('modelo_lumbar.h5')