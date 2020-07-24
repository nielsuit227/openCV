import os
import cv2 as cv
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt



height = 360
width = 160
size = (width, height)
for file in os.listdir('images'):
    fig, ax = plt.subplots(2, 3)
    img = cv.imread('images/' + file)
    img = cv.resize(img, size)

    # Fix intensity
    bilateral = cv.bilateralFilter(img, 15, 100, 80)            # Gaussian filter on intensity  (size, sigmaColor, simgaSpace)
    rescaled = exposure.rescale_intensity(img)                  # Intensity rescaled, og distribution
    equalHist = exposure.equalize_hist(img)                     # Intensity to uniform
    adaptHist = exposure.equalize_adapthist(img)                # Histogram Equalization over local area
    gammaCorrect = exposure.adjust_gamma(img, 0.5)                 # Gamma Correction (Power Law)
    logCorrect = exposure.adjust_log(img, 1)                       # Logarithmic Correction

    # Plot
    ax[0, 0].imshow(bilateral)
    ax[0, 0].set_title('Bilateral Filter')
    ax[0, 1].imshow(rescaled)
    ax[0, 1].set_title('Rescaled')
    ax[0, 2].imshow(equalHist)
    ax[0, 2].set_title('Equalize Histogram')
    ax[1, 0].imshow(adaptHist)
    ax[1, 0].set_title('Adaptive Histogram')
    ax[1, 1].imshow(gammaCorrect)
    ax[1, 1].set_title('Gamma Correction')
    ax[1, 2].imshow(logCorrect)
    ax[1, 2].set_title('Log Correction')
    plt.show()