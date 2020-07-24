import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


width = 160
height = 360
thres = 20
fig, ax = plt.subplots(2, 5)
for iter, file in enumerate(os.listdir('images')):
    img = cv.imread('images/'+file)
    img = cv.resize(img, (width, height))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 3)
    edges = cv.Canny(th3, 500, 1000)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    centers = []
    for i, contour in enumerate(contours):
        # Find bounding boxes
        (x, y, w, h) = cv.boundingRect(contour)
        centers.append((int(x + w / 2), int(y + h / 2)))

    # Find rows / cols
    row = []
    col = []
    first = True
    for center in centers:
        if first:
            col.append(center[0])
            row.append(center[1])
            first = False
        if np.min(abs(np.array(col) - center[0])) > thres:
            col.append(center[0])
        if np.min(abs(np.array(row) - center[1])) > thres:
            row.append(center[1])
    row = np.sort(np.array(row))
    col = np.sort(np.array(col))
    # Finally assign row/col indices to contours
    ind = []
    for center in centers:
        ind.append([np.where(abs(col - center[0]) == np.min(abs(col - center[0])))[0][0],
                    np.where(abs(row - center[1]) == np.min(abs(row - center[1])))[0][0]])
        cv.circle(img, center, 5, (255, 0, 0), -1)
        cv.putText(img, str(ind[-1]), (col[ind[-1][0]] - 25, row[ind[-1][1]] - 25), cv.FONT_HERSHEY_TRIPLEX, 0.5,
                   (255, 255, 255))

    j = np.remainder(iter, 5)
    ax[int(iter / 5), j].imshow(img)
    ax[int(iter / 5), j].set_title(file)
plt.show()