import cv2
import numpy
import scipy.cluster.vq as vq


def computeSIFT(filename, mask=False):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT(contrastThreshold=0.06)
    if (mask):
        (h, w) = img.shape[:2]
        (cX, cY) = ((int)(w * 0.5), (int)(h * 0.5))
        (axesX, axesY) = ((int)((w * 0.8) / 2), (int)((h * 0.8) / 2))
        ellipMask = numpy.zeros(img.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        kp, des = sift.detectAndCompute(gray, ellipMask, None)
    else:
        kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = numpy.histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

