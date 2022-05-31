import os
import sys
sys.path.insert(1, '../Utils/')
from settings import *
from label_matching import label_match
import matplotlib.pyplot as plt
# from matplotlib.image import imread

def visualize_dataset(show=True):
    folders = os.listdir(TRAIN_DIR)
    cl_name, _ = label_match()
    train_number = []
    class_num = []
    for folder in folders:
        train_files = os.listdir(TRAIN_DIR + folder)
        train_number.append(len(train_files))
        class_num.append(cl_name[int(folder)])
    zipped_lists = zip(train_number, class_num)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    train_number, class_num = [ list(tuple) for tuple in  tuples]
    plt.figure(figsize=(10,5))  
    plt.bar(class_num, train_number)
    plt.xticks(class_num, rotation='vertical')
    if show == True:
        plt.show()

    return plt

def print_random_images(show=True):
    pass

if __name__ == '__main__':
    print('Data Visualization')
    visualize_dataset()