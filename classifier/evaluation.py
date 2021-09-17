from os import listdir
from os.path import join
from sklearn.externals import joblib
import cv2
from feature.hog_feature import Hog_descriptor
import numpy as np

clf = joblib.load('Training')


def predict(img, clf):
    img = cv2.resize(img, (128, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    feature_vector = Hog_descriptor(img).extract()
    feature_vector = np.asarray(feature_vector)
    feature_vector = feature_vector.reshape((1, feature_vector.shape[0]))
    return clf.predict(feature_vector)


def evalution(posValid, negValid, clf):
    accuracy = 0
    for imagePath in listdir(posValid):
        img = cv2.imread(join(posValid, imagePath))
        if int(predict(img, clf)[0]) == 1:
            accuracy += 1
    for imagePath in listdir(negValid):
        img = cv2.imread(join(negValid, imagePath))
        if int(predict(img, clf)[0]) == 0:
            accuracy += 1
    return float(accuracy) / (float(len(listdir(posValid)) + len(listdir(negValid))))


pos = '/home/mrdung/PycharmProjects/hog/data/test/positive'
neg = '/home/mrdung/PycharmProjects/hog/data/test/negative'
print (evalution(pos, neg, clf))
