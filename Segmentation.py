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
for file in os.listdir('images'):
    fig, ax = plt.subplots(1, 4)
    img = cv.imread('images/' + file)
    img = cv.resize(img, size)

    # Fix intensity
    img = exposure.rescale_intensity(img)
    kernel = np.ones((9, 9), np.uint8)
    img = cv.dilate(cv.erode(img, kernel), kernel)

    # Low Level Segmentation
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
                         start_label=1)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = sobel(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    # Plot
    ax[0].imshow(mark_boundaries(img, segments_fz))
    ax[0].set_title('FelzenSzwalb')
    ax[1].imshow(mark_boundaries(img, segments_slic))
    ax[1].set_title('Simple Linear Iterative Clustering')
    ax[2].imshow(mark_boundaries(img, segments_quick))
    ax[2].set_title('QuickShift')
    ax[3].imshow(mark_boundaries(img, segments_watershed))
    ax[3].set_title('Watershed')
    plt.show()