import sys
sys.path.insert(1, '../Utils/')
from settings import *
from label_matching import label_match
from load_dataset import DataLoader
from custom_model import customModel
from tensorflow import keras
# from tensorflow.keras.optimizers import Adam, SGD


def main():
    classes_text, classes_images = label_match()
    data_loader = DataLoader()
    data_loader.prepare_test_data()

    model_1 = customModel(model_id = 1)
    model_1.prepare_to_train()
    tr_d, tr_l, t_d, t_l = data_loader.return_data()
    model_1.train_model(tr_d, tr_l, t_d, t_l)


if __name__ == '__main__':
    print('Create Model Module')
    main()