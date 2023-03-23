import os
import cv2 as cv
import numpy as np
from utils import *

h, w = 450, 450

img = cv.imread("aug/_3_984526.jpeg")
img, img_thresh = preprocess(h, w, img)
contours, hierarchy = cv.findContours(255-img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS

largest = findLargestArea(contours)

if largest.size != 0:
    largest = reorderPoints(largest)   
    # print(largest)

    # cv.drawContours(img, largest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    ortho = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    H1, _ = cv.findHomography(srcPoints=largest, dstPoints=ortho)
    img_warp = cv.warpPerspective(img, H1, (450, 450))

    boxes = splitBoxes(img_warp)

    model = load_model('model.h5')

    nums = []
    for box in boxes:
        img = np.asarray(box)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)

        predictions = model.predict(img)
    