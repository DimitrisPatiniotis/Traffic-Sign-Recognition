from settings import *
from PIL import Image
from sklearn.metrics import accuracy_score
import pandas as pd
import cv2
from sklearn.metrics import classification_report

def prepare_img(img_png):
    pil_img = Image.fromarray(img_png, 'RGB')
    resized_img = pil_img.resize((IMG_HEIGHT, IMG_WIDTH))
    return resized_img

def run_test_png(img_png, model):
    return model.predict_classes(prepare_img(img_png))

def test_data_predictions(model):
    test_csv = pd.read_csv(CSV_DIR + 'Test.csv')
    images = test_csv["Path"].values
    y_test = test_csv["ClassId"].values
    X_test = []
    for i in images:
        image = cv2.imread(CSV_DIR + '/' + i)
        X_test.append(prepare_img(image))
    X_test = np.array(X_test)
    predictions = model.predict_classes(X_test)
    print('Test Data accuracy: {}%'.format(round(accuracy_score(y_test, predictions)*100,2)))
    print(classification_report(y_test, predictions))
    return

if __name__ == '__main__':
    print('Run Tests util')