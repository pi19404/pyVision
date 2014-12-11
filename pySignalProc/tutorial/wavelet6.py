# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 12:39:27 2014

@author: pi19404
"""

from pySignalProc.pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title
from scipy import interpolate
from tabulate import tabulate
from scipy import integrate

from cvxopt import matrix, solvers
from pySignalProc.wavelet.pywtUtils import *

from pySignalProc.PieceWisePolynomial import *


from scipy.signal import (freqz, butter, bessel, cheby1, cheby2, ellip, 
                              tf2zpk, zpk2tf, lfilter, buttap, bilinear, cheb2ord, cheb2ap
                              )
from numpy import asarray, tan, array, pi, arange, cos, log10, unwrap, angle

from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from collections import defaultdict




def amplitude_complementary(h):
    z=np.zeros(len(h))
    z[0]=1
    return z-h;
    
    
    
    

figure()
h=np.array([0.2,0.4,0.6,0,0.8,0,0.2,0,0.4],float) 
h=np.append(h,np.zeros(1000))
Utils.plotSpectrum(h,1,None,False,False)
figure()


h2=np.copy(h)
h2[1]=0
Utils.plotSpectrum(h2,1,None,False,False)
figure()

g=amplitude_complementary(h)
Utils.plotSpectrum(g,1,None,False,False)
figure()
index=np.array(range(len(g)))
r=h-h*np.exp(1j*math.pi*index)
r=r.real
Utils.plotSpectrum(r,1,None,False,False)

show()









