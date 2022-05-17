# Setting up some general settings
import numpy as np

# Path Settings
BASE_DIR = '../'
META_DIR = BASE_DIR + 'Data/Meta/'
TEST_DIR = BASE_DIR + 'Data/Test/'
TRAIN_DIR = BASE_DIR + 'Data/Train/'
CSV_DIR = BASE_DIR + 'Data/'

# Image Resize Settings
IMG_RSZ_H = 30
IMG_RSZ_W = 30
IMG_RSZ_CHNLS = 3

# Setting total number of sign classes
TOTAL_CLASSES = 43

# Set Random Seed
def set_seed(num=42):
    np.random.seed(num)

if __name__ == '__main__':
    print('General Settings Util')