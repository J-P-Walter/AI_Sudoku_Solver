import os
import cv2 as cv
import numpy as np
from utils import *
import tensorflow

h, w = 450, 450

img = cv.imread("aug/_8_607650.jpeg")

img, img_thresh = preprocess(h, w, img)
contours, hierarchy = cv.findContours(255-img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS

largest = findLargestArea(contours)

if largest.size != 0:
    largest = reorderPoints(largest)   
    # print(largest)

    # cv.drawContours(img, largest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    # cv.imshow('a', img)
    # cv.waitKey(0)
    
    ortho = np.float32([[0, 0], [504, 0], [0, 504], [504, 504]])
    H1, _ = cv.findHomography(srcPoints=largest, dstPoints=ortho)
    img_warp = cv.warpPerspective(img, H1, (504, 504))
    img_warp_gray = cv.cvtColor(img_warp, cv.COLOR_BGR2GRAY)
    boxes = splitBoxes(img_warp_gray)

    model = tensorflow.keras.models.load_model('test.h5')

    nums = []
    for box in boxes:
        box = box[4:box.shape[0] - 4, 4:box.shape[1] -4]
        box = cv.bitwise_not(box)
        box = cv.resize(box, (28, 28))
        box = box / 255
        cv.imshow('a', box)
        cv.waitKey(0)
        box = box.reshape(1, 28, 28, 1)     


        predictions = model.predict(box)
        print(predictions)
        classes=np.argmax(predictions,axis=1)
        print(classes)
        
    print(nums)
