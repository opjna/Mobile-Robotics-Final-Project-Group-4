import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from scipy.stats import multivariate_t
from convolution import convolution

def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )/(sigma*np.sqrt(2*np.pi))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def attempt_skewed2D(shape=(3,3),upsilon=2):
    """
    2D skewed mask - attempt
    """
    m1,n1 = [(ss-1.)/2. for ss in shape]
    y1,x1 = np.mgrid[0:2*m1+1:1, 0:2*n1+1:1]
    pos = np.dstack((x1, y1))
    
    rv = multivariate_t([1*(2*m1+1)/5, 3*(2*n1+1)/5],[[1, 0], [0, 1]], df=upsilon)
    tile = np.array(rv.pdf(pos))
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = tile
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussian_blur(image, verbose=False):
    kernel = gauss2D((5,5),1)
    return convolution(image, kernel, average=False, verbose=verbose)

def skew_blur(image, verbose=False):
    kernel = attempt_skewed2D((5,5),1)
    return convolution(image, kernel, average=False, verbose=verbose)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())