import pickle
from os import listdir
from os.path import join

import cv2

from feature.hog_feature import Hog_descriptor


def extract_image(img):
    img = cv2.resize(img, (128, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    vector = Hog_descriptor(img).extract()
    return vector


def data_to_feature(dataPath):
    result = []
    for imagePath in listdir(dataPath):
        img = cv2.imread(join(dataPath, imagePath))
        if img is not None:
            result.append(extract_image(img))
    return result


dataPath = '/home/mrdung/PycharmProjects/hog/data/positive'
result = data_to_feature(dataPath)
with open("positive_feature.txt", "wb") as fp:  # Pickling
    pickle.dump(result, fp)
print ("Positive Done")
dataPath = '/home/mrdung/PycharmProjects/hog/data/negative'
result = data_to_feature(dataPath)
with open("negative_feature.txt", "wb") as fp:  # Pickling
    pickle.dump(result, fp)
print ("Done")
