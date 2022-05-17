import sys
sys.path.insert(1, '../Utils/')
from settings import *
from label_matching import label_match
from load_dataset import DataLoader
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD

def main():
    classes_text, classes_images = label_match()
    data_loader = DataLoader()
    data_loader.prepare_test_data()

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

    lr = 0.001
    epochs = 10

    opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
    # opt = SGD(lr=lr, momentum=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(data_loader.training_data, data_loader.training_labels, batch_size=32, epochs=epochs, validation_data=(data_loader.test_data, data_loader.test_labels))

if __name__ == '__main__':
    print('Create Model Module')
    main()