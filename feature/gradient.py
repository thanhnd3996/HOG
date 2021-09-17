import cv2
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='image')
    parse.add_argument('--image', help='input_image')
    args = parse.parse_args()
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(gx)
    fig.add_subplot(1, 2, 2)
    plt.imshow(gy)
    plt.show()
