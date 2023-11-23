import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Leitura do dataset, imagens e metadados
dataset_dir = "Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"
df = pd.read_csv("Chest_xray_Corona_Metadata.csv")

# Dividindo dois conjuntos de treinamneto em dados de train e test
train_data = df[df['Dataset_type'] == 'TRAIN']
test_data = df[df['Dataset_type'] == 'TEST']
train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data['Label'])

# Configurações para pré-processamento da imagem e aumento de dados
image_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(
    # facilitar a convergência durante o treinamento de redes neurais.
    rescale=1./255,
    # distorção de cisalhamento aplicada às imagens
    shear_range=0.2,
    #amplia ou reduz imagem(0,9 (zoom out) a 1,1 (zoom in).)
    zoom_range=0.2,
    #espelha horizontalmente
    horizontal_flip=True
)

#normaliza os dados para o pixel ficar com valores entre 0 e 1
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Carregamento dos dados
#responsável por fornecer lotes de dados de treinamento
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=os.path.join(dataset_dir, "train"),
    x_col="X_ray_image_name",
    #rotulos relacionados as imagens
    y_col="Label",
    target_size=image_size,
    #tamanho dos lotes, afeta o número de amostras de treinamento consideradas antes de uma atualização de peso
    batch_size=batch_size,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=os.path.join(dataset_dir, "train"),
    x_col="X_ray_image_name",
    y_col="Label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=os.path.join(dataset_dir, "test"),
    x_col="X_ray_image_name",
    y_col="Label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary"
)

# Construção do modelo
model = Sequential()
#A função ReLU é definida como f(x) = max(0, x), o que significa que ela retorna zero para valores negativos e retorna o próprio valor para valores não negativos. 
# ajuda a introduzir não linearidades na rede, permitindo que ela aprenda padrões mais complexos nos dados.

#Camadas Conv2D são usadas para extrair características das imagens de entrada
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
#transformar os mapas de features bidimensionais em um vetor unidimensional
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#transforma qualquer valor real em um intervalo entre 0 e 1
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks para treinamento
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Treinamento do modelo
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    # 1- interrompe o treinamento se a loss não diminuir após um número específico de epochs
    # 2 - Salva o modelo com os melhores pesos (menor perda de validação) durante o treinamento
    callbacks=[early_stopping, model_checkpoint]
    # .fit retorna um objeto History, que contém informações sobre as métricas de treinamento e validação ao longo das épocas.
)

# Avaliação no conjunto de teste
test_loss, test_acc = model.evaluate(test_generator)
print(f"Acurácia no conjunto de teste: {test_acc}")

# Plotagem de métricas de treinamento
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
