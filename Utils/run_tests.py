from settings import *
from PIL import Image
from sklearn.metrics import accuracy_score
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report

def prepare_img(img_png):
    pil_img = Image.fromarray(img_png, 'RGB')
    
    resized_img = pil_img.resize((IMG_RSZ_H, IMG_RSZ_W))
    return resized_img

def run_test_png(img_png, model):
    return model.predict_classes(prepare_img(img_png))

def test_data_predictions(model):
    test_csv = pd.read_csv(CSV_DIR + 'Test.csv')
    images = test_csv["Path"].values
    y_test = test_csv["ClassId"].values
    print(y_test[0])
    X_test = []
    for i in images:
        curr_path = TEST_DIR + str(i)
        curr_image = cv2.imread(curr_path)
        image_fromarray = Image.fromarray(curr_image, 'RGB')
        resized_image = image_fromarray.resize((IMG_RSZ_H, IMG_RSZ_W))
        X_test.append(np.array(resized_image))
    X_test = np.array(X_test)
    predictions = model.predict(X_test)
    predictions = np.array([list(i).index(max(i)) for i in predictions])
    print(y_test[:2], predictions[:2])
    print('Test Data accuracy: {}%'.format(round(accuracy_score(y_test, predictions)*100,2)))
    print(classification_report(y_test, predictions))
    return

if __name__ == '__main__':
    print('Run Tests util')