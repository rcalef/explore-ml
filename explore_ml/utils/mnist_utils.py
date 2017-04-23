#! /usr/bin/env python3

import gzip
import os

import numpy as np
from .data_utils import read_int

def read_mnist_images(filename, normalize = False):
    fh = gzip.open(filename,"rb")
    fh.seek(4)
    
    nimages = read_int(fh,4)
    nrows = read_int(fh,4)
    ncols = read_int(fh,4)
    npixels = nrows * ncols
    
    print("Loading %d %dx%d images from file: %s" % 
          (nimages,nrows,ncols,os.path.basename(filename)))
    
    images = np.empty((nimages,nrows,ncols)) 
    for i in range(nimages):
        if normalize:
            images[i,:,:] = (np.fromstring(fh.read(npixels),dtype=np.uint8).reshape((28,28)) - 255/2) / 255
        else:
            images[i,:,:] = np.fromstring(fh.read(npixels),dtype=np.uint8).reshape((28,28))

    return images

def read_mnist_labels(filename):
    fh = gzip.open(filename,"rb")
    fh.seek(4)
    
    nlabels = read_int(fh,4)
    print("Loading %d labels from file: %s" % 
          (nlabels,os.path.basename(filename)))
    
    labels = np.empty((nlabels,))
    for i in range(nlabels):
        labels[i] = read_int(fh,1,signed=False)
    return labels
