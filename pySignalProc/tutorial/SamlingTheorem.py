# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 20:26:55 2014

@author: pi19404
"""

from pyCommon import *
from numpy import linspace,cos,pi,ceil,floor,arange
from pylab import plot,show,axis


# sampling a signal badlimited to 40 Hz 
# with a sampling rate of 800 Hz
f = 40;  # Hz
tmin = -1;
tmax = 1;
t = linspace(tmin, tmax, tmax*500);
x =  cos(2*pi*f*t); # signal sampling

plt.figure(1)
title('cosine signal spectrum')
Utils.plotSpectrum(x,500)
plt.grid()

x=Utils.pulsetran(f,2000)

plt.figure(2)

Utils.plotSpectrum(x,1)
title('pulse train spectrum')


x=numpy.sinc(f*t)
x=x
plt.figure(3)
Y,freq=Utils.plotSpectrum(x,500,'sinc function')


Y=abs(Y)
Y1=np.zeros(Y.shape)
for i in range(-2,3):
    Y1=Y1+Utils.delay(Y,i*80)

plt.figure(4)
plt.plot(freq,Y1)
title('sampling at 80Hz')


plt.figure(5)
Y,freq=Utils.plotSpectrum(x,500)

Y=abs(Y)
Y1=np.zeros(Y.shape)
for i in range(-2,3):
    Y1=Y1+Utils.delay(Y,i*60)
    
plt.figure(5)
plt.clf()
plt.plot(freq,Y1)
title('sampling at 60Hz')
plt.grid()

f=40
tmin = -1;
tmax = 1;
t = linspace(tmin, tmax, tmax*600);
x=numpy.sinc(f*t/2)
x=x*x

x=Utils.downsample(x,10)

plt.figure(6)
Y,freq=Utils.plotSpectrum(x,60)
plt.grid()

print freq,Y

show()