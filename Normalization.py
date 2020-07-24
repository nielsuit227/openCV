import os
import cv2 as cv
import numpy as np
from skimage import exposure
from skimage import data, io, segmentation, color
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, mark_boundaries, watershed
from skimage.future import graph
import matplotlib.pyplot as plt



height = 360
width = 160
size = (width, height)
fig, ax = plt.subplots(2, 5)
for i, file in enumerate(os.listdir('images')):
    img = cv.imread('images/' + file)
    img = cv.resize(img, size)

    # Fix intensity
    # kernel = np.ones((9, 9), np.uint8)
    # img = cv.dilate(cv.erode(img, kernel), kernel)

    # Quite a lot of white icons, let's select them first --> WORKS REALLY WELL
    ret, plain_tiles = cv.threshold(img, 250, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    marker = cv.dilate(cv.erode(plain_tiles, kernel), kernel)
    kernel = np.ones((9, 9), np.uint8)
    marker = cv.erode(cv.dilate(marker, kernel), kernel)
    ax[int(i / 5)][i%5].imshow(marker)
plt.show()

    # centers = []
    # for i, contour in enumerate(contours):
    #     # Find bounding boxes
    #     (x, y, w, h) = cv.boundingRect(contour)
    #
    #     # New filters, based on sizes
    #     if w < width / 7 or w > width / 3:
    #         continue
    #     elif h < height / 14 or h > height / 5:
    #         continue
    #     else:
    #         # Store and draw centers
    #         centers.append((int(x + w / 2), int(y + h / 2)))
    #         cv.circle(img, centers[-1], 5, (255, 0, 0), -1)
    #
    # # Find rows / cols
    # thres = 50
    # row = []
    # col = []
    # first = True
    # for center in centers:
    #     if first:
    #         col.append(center[0])
    #         row.append(center[1])
    #         first = False
    #     if np.min(abs(np.array(col) - center[0])) > thres:
    #         col.append(center[0])
    #     if np.min(abs(np.array(row) - center[1])) > thres:
    #         row.append(center[1])
    # row = np.sort(np.array(row))
    # col = np.sort(np.array(col))
    # # Finally assign row/col indices to contours
    # ind = []
    # for center in centers:
    #     ind.append([np.where(abs(col - center[0]) == np.min(abs(col - center[0])))[0][0],
    #                 np.where(abs(row - center[1]) == np.min(abs(row - center[1])))[0][0]])
    #     cv.putText(img, str(ind[-1]), (col[ind[-1][0]] - 25, row[ind[-1][1]] - 25), cv.FONT_HERSHEY_TRIPLEX, 0.5,
    #                (255, 255, 255))
    #
    # cv.imshow(file, img)

