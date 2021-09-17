import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while (True):
        count = 50
        ret, frame = cap.read()
        if count == 50:
            (rects, weight) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(16, 16), scale=1.05)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            count = 0
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
