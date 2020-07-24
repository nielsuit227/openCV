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
ksize_open = [1, 3, 5, 7]      # Window size dilate / erode
size = (width, height)
for file in os.listdir('images'):
    fig, ax = plt.subplots(1, 4)
    # Read Image
    img = cv.imread('images/'+file)

    # Resize
    img = cv.resize(img, size)

    # Brightness Correction
    img = exposure.rescale_intensity(img)

    for j in range(4):
        # Opening
        kernel = np.ones((ksize_open[j], ksize_open[j]), np.uint8)
        temp = cv.dilate(cv.erode(img, kernel), kernel)

        # Region Adjacency Graph
        labels = segmentation.slic(temp, compactness=25, n_segments=250, start_label=1)
        g = graph.rag_mean_color(temp, labels)
        labels2 = graph.merge_hierarchical(labels, g, thresh=45, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)
        for ii in range(np.max(labels2)):
            if np.logical_or(np.sum(labels2 == ii) > 1000, np.sum(labels2 == ii) < 150):
                labels2[labels2 == ii] = 0
        out = color.label2rgb(labels2, temp, kind='avg', bg_label=0)
        out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
        ax[j].imshow(out)
        ax[j].set_title('K = %i' % ksize_open[j])

    plt.show()