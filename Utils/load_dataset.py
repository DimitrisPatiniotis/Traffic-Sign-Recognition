import sys
import os
from PIL import Image
import cv2
sys.path.insert(1, '../Utils/')
from settings import *
from tqdm import tqdm
from sklearn.model_selection import  train_test_split
from tensorflow import keras

class DataLoader():
    def __init__(self, test_data_dir = TEST_DIR, train_data_dir = TRAIN_DIR, test_size=0.3):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.test_size = test_size
        self.training_data = np.empty([0])
        self.training_labels = np.empty([0])
        self.test_data = np.empty([0])
        self.test_labels = np.empty([0])

    def load_training_data(self, resize=True):
        self.training_data = []
        self.training_labels = []
        print('Loading Training Data')
        for i in tqdm(range(TOTAL_CLASSES)):
            curr_path = TRAIN_DIR + str(i)
            images = os.listdir(curr_path)
            for image in images:
                try:
                    curr_image = cv2.imread(curr_path + '/' + image)
                    image_fromarray = Image.fromarray(curr_image, 'RGB')
                    resized_image = image_fromarray.resize((IMG_RSZ_H, IMG_RSZ_W))
                    self.training_data.append(np.array(resized_image))
                    self.training_labels.append(i)
                except:
                    print(curr_image + ' did not load properely')
        self.training_data =  np.array(self.training_data)
        self.training_labels = np.array(self.training_labels)

    def shuffle_training_data(self):
        shuffle_indexes = np.arange(self.training_data.shape[0])
        np.random.shuffle(shuffle_indexes)
        self.training_data = self.training_data[shuffle_indexes]
        self.training_labels = self.training_labels[shuffle_indexes]
    
    def split_training_data(self):
        # Create Split
        X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_labels, test_size= self.test_size , random_state=42, shuffle=True)
        self.training_data, self.training_labels, self.test_data, self.test_labels = X_train, y_train, X_test, y_test
    
    def one_hot_labels(self):
        self.training_labels, self.test_labels = keras.utils.to_categorical(self.training_labels, 43), keras.utils.to_categorical(self.test_labels, 43)

    def prepare_test_data(self):
        self.load_training_data(self)
        self.shuffle_training_data()
        self.split_training_data()
        self.one_hot_labels()

    def return_data(self):
        return self.training_data, self.training_labels, self.test_data, self.test_labels

    
if __name__ == '__main__':
    print('Data Loader Util')