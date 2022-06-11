
import sys
sys.path.insert(1, '../Utils/')
from settings import *
from label_matching import label_match
from load_dataset import DataLoader
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from models import *


class customModel():
    def __init__(self, name = 'Custom Model', model_id = 1, learning_rate = 0.001, optimizer = 'Adam', epochs = 20, loss = 'categorical_crossentropy', metrics=['accuracy']):
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.loss = loss
        self.metrics_list = metrics
        self.model_id = model_id
        self.optimizer = None
        self.model = None
    
    # import model from models.py
    def define_model(self):
        if self.model_id == 1:
            self.model = model_1()
        if self.model_id == 2:
            self.model = model_2()

    def set_optimizer(self):
        if self.optimizer_name == 'Adam':
            self.optimizer = Adam(lr=self.learning_rate, decay=(self.learning_rate / (self.epochs * 0.5)))

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics_list)
    
    def prepare_to_train(self):
        self.define_model()
        self.set_optimizer()
        self.compile_model()

    def train_model(self, training_data, training_labels, test_data, test_labels, bs=32):
        self.model.fit(training_data, training_labels, batch_size=bs, epochs=self.epochs, validation_data=(test_data, test_labels))

if __name__ == '__main__':
    print('Create Model Module')