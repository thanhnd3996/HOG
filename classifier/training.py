import pickle
from sklearn import svm

import numpy as np
from sklearn.externals import joblib


def training():
    with open('./classifier/positive_feature.txt', 'r') as pf:
        positive = pickle.load(pf)
    with open('./classifier/negative_feature.txt', 'r') as nf:
        negative = pickle.load(nf)
    X = np.concatenate((positive, negative), axis=0)
    labels = np.concatenate((np.ones((len(positive), 1)), np.zeros((len(negative), 1))), axis=0)
    clf = svm.SVC(kernel='linear', C=0.03)
    clf.fit(X, labels)
    joblib.dump(clf, 'Training')
    print(0)


if __name__ == '__main__':
    training()
