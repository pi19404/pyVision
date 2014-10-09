# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:26:41 2014

@author: pi19404
"""
import pylab
from PIL import Image
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import numpy
rng = numpy.random.RandomState(23455)

im = Image.open(open('/home/pi19404/repository/pyVision/data/t1.jpg'))
#im /= 255.   # normalise to 0-1, it's easier to work in float space
im = numpy.asarray(im, dtype='float64') / 256.
# make some kind of kernel, there are many ways to do this...

w_shp = (9, 9,3)
w_bound = numpy.sqrt(3 * 9 * 9)
kernel=numpy.asarray(rng.uniform(low=-1.0 / w_bound,high=1.0 / w_bound,size=w_shp),dtype=im.dtype)
#kernel = t.reshape(21, 1) * t.reshape(1, 21
#kernel = np.asarray(rng.uniform(-0.1,0.1,size=(21,21)),dtype='float64')
#kernel /= kernel.sum()   # kernel should sum to 1!  :) 

# convolve 2d the kernel with each channel
r = scipy.signal.convolve2d(im[:,:,0], kernel[:,:,0], mode='same')
g = scipy.signal.convolve2d(im[:,:,1], kernel[:,:,1], mode='same')
b = scipy.signal.convolve2d(im[:,:,2], kernel[:,:,2], mode='same')

# stack the channels back into a 8-bit colour depth image and plot it
im_out = np.dstack([r, g, b])
#im_out = (im_out ).astype(np.uint8) 
img2 = scipy.ndimage.convolve(im, kernel, mode='constant', cval=0.0)

pylab.subplot(1,3,1)
pylab.imshow(im)
pylab.gray();

pylab.subplot(1,3,2)
pylab.imshow(im_out[:,:,0])
pylab.subplot(1,3,3)
pylab.imshow(im_out[:,:,1])
pylab.gray();

pylab.show()