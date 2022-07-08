from tensorflow import keras
from settings import *

def model_1():
    model = keras.models.Sequential([    
        keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_RSZ_H,IMG_RSZ_W,IMG_RSZ_CHNLS)),
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(43, activation='softmax')
    ])
    return model

def model_2():
    model = keras.models.Sequential([    
        keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_RSZ_H,IMG_RSZ_W,IMG_RSZ_CHNLS)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(43, activation='softmax')
    ])
    return model