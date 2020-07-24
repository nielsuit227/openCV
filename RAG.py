import os
import cv2 as cv
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.future import graph
from skimage import segmentation, color


def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


height = 360        # Height of resized images
width = 160         # Width of resized images
thres = 25          # Minimum distance between rows / columns
ksize_open = 9      # Window size dilate / erode
size = (width, height)
fig, ax = plt.subplots(2, 5)
for i, file in enumerate(os.listdir('images')):
    # Read Image
    img = cv.imread('images/'+file)

    # Resize
    img = cv.resize(img, size)

    # Brightness Correction
    img = exposure.rescale_intensity(img)

    # Region Adjacency Graph
    labels = segmentation.slic(img, compactness=25, n_segments=250, start_label=1)
    g = graph.rag_mean_color(img, labels)
    labels2 = graph.merge_hierarchical(labels, g, thresh=45, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    for ii in range(np.max(labels2)):
        if np.logical_or(np.sum(labels2 == ii) > 1000, np.sum(labels2 == ii) < 150):
            labels2[labels2 == ii] = 0
    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    out = np.uint8(out * 255)

    centers = []
    for ii in range(np.max(labels2) - 1):
        xInd, yInd = np.where(labels2 == ii + 1)
        if len(xInd) == 0:
            continue
        xCenter = int(np.mean(xInd))
        yCenter = int(np.mean(yInd))
        w = max(xInd) - min(xInd)
        h = max(yInd) - min(yInd)

        # New filters, based on sizes
        # if w < 20 or w > 40:
        #     continue
        # elif h < 20 or h > 40:
        #     continue
        # elif abs(w - h) > 20:
        #     continue
        # else:
            # Store and draw centers
        centers.append((yCenter, xCenter))
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

    ax[int(i / 5)][i%5].imshow(img)
    ax[int(i / 5)][i%5].set_title(file)
plt.show()