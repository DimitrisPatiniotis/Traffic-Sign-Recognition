
from cProfile import label
import sys
sys.path.insert(1, '../Utils/')
from settings import *
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from models import *


class customModel():
    def __init__(self, name = 'Custom Model', model_id = 1, learning_rate = 0.001, optimizer = 'Adam', epochs = 20, loss = 'categorical_crossentropy', metrics=['accuracy'], callback = None, model_patience = 5):
        self.name = name
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.loss = loss
        self.metrics_list = metrics
        self.model_id = model_id
        if callback == 'early_stopping':
            self.callback = EarlyStopping(monitor='loss', patience=model_patience)
        else:
            self.callback = None
        self.optimizer = None
        self.model = None
        self.history_info = None
    
    # import model from models.py
    def define_model(self):
        if self.model_id == 1:
            self.model = model_1()
        if self.model_id == 2:
            self.model = model_2()

    def set_optimizer(self):
        if self.optimizer_name == 'Adam':
            self.optimizer = Adam(learning_rate=self.learning_rate, decay=(self.learning_rate / (self.epochs * 0.5)))
        elif self.optimizer_name == 'SGD':
            self.optimizer = SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Adadelta':
            self.optimizer = Adadelta(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-07)

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics_list)
    
    def prepare_to_train(self):
        self.define_model()
        self.set_optimizer()
        self.compile_model()

    def train_model(self, training_data, training_labels, test_data, test_labels, bs=32):
        if self.callback == None:
            history = self.model.fit(training_data, training_labels, batch_size=bs, epochs=self.epochs, validation_data=(test_data, test_labels))
            self.history_info = history.history
        else:
            history = self.model.fit(training_data, training_labels, batch_size=bs, epochs=self.epochs, validation_data=(test_data, test_labels), callbacks=[self.callback])
            self.history_info = history.history

    def graph_history_accuracy(self):
        try:
            plt.plot(self.history_info['accuracy'], label='Train Accuracy')
            plt.plot(self.history_info['val_accuracy'], label='Validation Accuracy')
            plt.title('Train & Validation Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()
        except:
            print('Error while trying to display model accuracy history')

    def graph_history_loss(self):
        try:
            plt.plot(self.history_info['loss'], label="Train Loss")
            plt.plot(self.history_info['val_loss'], label="Validation Loss")
            plt.title('Train & Validation Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()
        except:
            print('Error while trying to display model loss history')

if __name__ == '__main__':
    print('Create Model Module')