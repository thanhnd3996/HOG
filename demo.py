import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

if __name__ == '__main__':
    img = cv2.imread('test.JPG')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weight) = hog.detectMultiScale(img, winStride=(4, 4), padding=(16, 16), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
