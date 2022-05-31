import os
import sys
sys.path.insert(1, '../Utils/')
from settings import *
from label_matching import label_match
from random import choice
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd

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

def print_sample_images(show=True):
    img_paths = pd.read_csv(CSV_DIR + 'Test.csv')["Path"].values
    plt.figure(figsize=(25,25))

    for img_number in range(1, 16):
        plt.subplot(3,5,img_number)
        random_img_path = CSV_DIR + choice(img_paths)
        rand_img = imread(random_img_path)
        plt.imshow(rand_img)
        plt.grid()
        plt.xlabel(rand_img.shape[1], fontsize = 10)
        plt.ylabel(rand_img.shape[0], fontsize = 10)
    if show == True:
        plt.show()
    
    return plt

if __name__ == '__main__':
    print('Data Visualization')