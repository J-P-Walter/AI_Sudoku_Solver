import numpy as np
import cv2
from imutils import grab_contours
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from keras.utils import img_to_array
import tensorflow as tf
from sudoku import Sudoku


img = cv2.imread("aug/test.jpeg")
model = tf.keras.models.load_model('another_model.h5')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 3)
img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img_thresh = cv2.bitwise_not(img_thresh)


contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS

corners = np.array([])
max_area = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 50:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            corners = approx
            max_area = area

# cv2.drawContours(img, [corners], -1, (0, 255, 0), 2)
# cv2.imshow("Puzzle Outline", img)
# cv2.waitKey(0)

if not np.any(corners):
    raise Exception("Could not find puzzle")

puzzle = four_point_transform(img, corners.reshape(4, 2))
warped = four_point_transform(img_gray, corners.reshape(4, 2))
# cv2.imshow("Puzzle Transform", puzzle)
# cv2.waitKey(0)

stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

found_puzzle = [[0]*9 for i in range(9)]

for y in range(0, 9):
# initialize the current list of cell locations
    for x in range(0, 9):
        # compute the starting and ending (x, y)-coordinates of the
        # current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY
        cell = warped[startY:endY, startX:endX]
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #empty cell
        if len(contours) == 0:
            continue

        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # compute the percentage of masked pixels relative to the total
        # area of the image
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)
        # if less than 3% of the mask is filled then we are looking at
        # noise and can safely ignore the contour
        if percentFilled < 0.03:
            continue
        # apply the mask to the thresholded cell
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)


        if digit is not None:
            digit = cv2.resize(digit, (28,28))
            digit = digit.astype("float") / 255.0
            digit = img_to_array(digit)
            digit = np.expand_dims(digit, axis=0)

            prediction = model.predict(digit).argmax(axis=1)
            found_puzzle[y][x] = prediction[0]

for row in found_puzzle:
    print(row)
board = [[8, 0, 0, 0, 1, 0, 0, 0, 9],
[0, 5, 0, 8, 0, 7, 0, 1, 0],
[0, 0, 4, 0, 9, 0, 7, 0, 0],
[0, 6, 0, 7, 0, 1, 0, 2, 0],
[5, 0, 8, 0, 6, 0, 1, 0, 7],
[0, 1, 0, 5, 0, 2, 0, 9, 0],
[0, 0, 7, 0, 4, 0, 6, 0, 0],
[0, 8, 0, 3, 0, 9, 0, 4, 0],
[3, 0, 0, 0, 5, 0, 0, 0, 8]]
puzzle = Sudoku(3, 3, board=board)

print()
print("--------------------")
print()
puzzle.solve().show_full()


