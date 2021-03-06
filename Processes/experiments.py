from operator import mod
import sys
sys.path.insert(1, '../Utils/')
from settings import *
from run_tests import *
from label_matching import label_match
from load_dataset import DataLoader
from custom_model import customModel
from tensorflow import keras
from ml_baseline import SVM

def svm_experiment():
    data_loader = DataLoader()
    data_loader.prepare_test_for_svm()
    tr_data, tr_labels, ts_data, ts_labels = data_loader.return_data()
    svm = SVM()
    svm.run_svm_experiment(tr_data, tr_labels, ts_data, ts_labels)
    print('SVM accuracy is ' + str(round(svm.accuracy * 100, 2)))
    return svm

def architecture_experiment(data_loader):
    tr_data, tr_labels, ts_data, ts_labels = data_loader.return_data()
    models = []
    for i in range(1,3):
        curr_model = customModel(model_id = i, name = 'Architecture {}'.format(i))
        curr_model.prepare_to_train()
        curr_model.train_model(tr_data, tr_labels, ts_data, ts_labels)
        models.append(curr_model)
    test_data_predictions(curr_model.model)
    return models

def optimizer_experiment(data_loader):
    tr_data, tr_labels, ts_data, ts_labels = data_loader.return_data()
    models = []
    for i in ['Adam', 'SGD', 'Adadelta']:
        curr_model = customModel(model_id = 1, optimizer = i, name='{} Optimizer'.format(i))
        curr_model.prepare_to_train()
        curr_model.train_model(tr_data, tr_labels, ts_data, ts_labels)
        models.append(curr_model)
        test_data_predictions(curr_model.model)
    return models

def training_size_experiment():

    models = []
    for i in [5000, 10000, 20000]:
        data_loader = DataLoader(reduce_dataset = True, training_size=i)
        data_loader.prepare_test_data()
        tr_data, tr_labels, ts_data, ts_labels = data_loader.return_data()
        curr_model = customModel(model_id = 1, name='Model trained with {} samples'.format(i), epochs=20, callback = 'early_stopping', model_patience=4)
        curr_model.prepare_to_train()
        curr_model.train_model(tr_data, tr_labels, ts_data, ts_labels)
        models.append(curr_model)
        print(curr_model.name)
        test_data_predictions(curr_model.model)
    return models

def get_best_model(data_loader):
    tr_data, tr_labels, ts_data, ts_labels = data_loader.return_data()
    model = customModel(model_id = 1, name = 'Best Performing Model', epochs=100, callback='early_stopping', model_patience=10)
    model.prepare_to_train()
    model.train_model(tr_data, tr_labels, ts_data, ts_labels)
    model.graph_history_accuracy()
    test_data_predictions(curr_model.model)
    return model

def main():
    data_loader = DataLoader()
    data_loader.prepare_test_data()
    svm_experiment()
    architecture_experiment(loader)
    optimizer_experiment(data_loader)
    training_size_experiment()
    get_best_model(data_loader)

if __name__ == '__main__':
    main()