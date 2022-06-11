import sys
sys.path.insert(1, '../Utils/')
from settings import *
from label_matching import label_match
from load_dataset import DataLoader
from custom_model import customModel
from tensorflow import keras
from ml_baseline import SVM
# from tensorflow.keras.optimizers import Adam, SGD


def main():
    classes_text, classes_images = label_match()
    data_loader = DataLoader()
    data_loader.prepare_test_for_svm()
    tr_data, tr_labels, ts_data, ts_labels = data_loader.return_data()

    svm = SVM()
    svm.run_svm_experiment(tr_data, tr_labels, ts_data, ts_labels)
    print('SVM accuracy is ' + str(round(svm.accuracy * 100, 2)))

    data_loader.prepare_test_data()
    tr_data, tr_labels, ts_data, ts_labels = data_loader.return_data()
    model_1 = customModel(model_id = 1)
    model_1.prepare_to_train()
    model_1.train_model(tr_data, tr_labels, ts_data, ts_labels)


if __name__ == '__main__':
    print('Create Model Module')
    main()