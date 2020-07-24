import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data, exposure
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi


size = (160, 360)
for file in os.listdir('images'):
    img = cv.imread('images/'+file)
    img = cv.resize(img, size)
    rescaled = exposure.rescale_intensity(img)                  # Intensity rescaled, og distribution

    gray = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    # markers = np.ones_like(gray) + 1
    # for col in range(4):
    #     for row in range(8):
    #         markers[(row + 1) * 40, (col + 1) * 32] = 2
    # markers = cv.watershed(img, np.uint8(markers))
    # img[markers == -1] = [255, 0, 0]
    # plt.imshow(img)
    # plt.show()

    # denoise image
    denoised = rank.median(gray, disk(2))
    markers = rank.gradient(denoised, disk(5)) < 15
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(denoised, disk(1))
    labels = watershed(gradient, markers)
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img)
    ax[0].set_title('Original')
    ax[1].imshow(markers)
    ax[1].set_title('Markers')
    ax[2].imshow(gradient)
    ax[2].set_title('Gradient')
    ax[3].imshow(labels)
    ax[3].set_title('Labels')
    plt.show()
