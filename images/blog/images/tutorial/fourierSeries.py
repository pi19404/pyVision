# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 17:02:45 2014

@author: pi19404
"""

import numpy, scipy, pylab, random  
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np;
import cmath
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange


cexp=numpy.vectorize(cmath.exp);
cabs=numpy.vectorize(abs);

import cmath

def compute_dft(input):
    n = len(input)
    output = [complex(0)] * n
    for k in range(n):  # For each output element
        s = complex(0)
        for t in range(n):  # For each input element            
         #   if k==0:
          #      print input[t] * cmath.exp(-2j * cmath.pi * t * k / n)
            s += input[t] * cmath.exp(-2j * cmath.pi * t * k / n)
        output[k] = s
    return output
    
    
def FourierSeries(input,N=None):
    
    if N==None:
        N=len(input);
    #N=np.shape(input)[0];
    w=2*cmath.pi/N;
    input=input[0:N];
    n=numpy.arange(0,N);    
    #print n,w
    
    r=cexp(-1j*w*n);
    #print r
    output = [complex(0)] * N    
    for k in range(N):        
        r=input*cexp(-1j*w*n*k)     
        #if k==0:
            #print input*cexp(-1j*w*n*k)     
        output[k]=np.sum(r);
        
    return output;
    
    
Fs=150;
Ts=1.0/Fs;
xs=numpy.arange(0,1,Ts) #generate Xs (0.00,0.01,0.02,0.03,...,100.0)  
F=10;
F1=15;
omega=2*np.pi*F;
omega1=2*np.pi*F1;

signal = numpy.cos(omega*xs)+numpy.cos(omega1*xs)
signal1=signal[0:Fs/F];

r=FourierSeries(signal1)
r1=FourierSeries(signal)
#print r
a=cabs(r)
a1=cabs(r1)
#print a
subplot(2,1,1)
plot(xs,signal)
xlabel('Time')
ylabel('Amplitude')
title('time doman')
subplot(2,1,2)
#fr=np.arange(0,Fs,F);
#
#
#plt.stem(fr,a,'r') # plotting the spectrum
#xlabel('Freq (Hz)')
#ylabel('|Y(freq)|')
#title('single period')
#ticks=np.arange(0,Fs+1,F);
#plt.xticks(ticks,ticks);
#
#
#subplot(3,1,3)
fr=np.arange(0,Fs);


plt.stem(fr,a1,'r') # plotting the spectrum
xlabel('Freq (Hz)')
ylabel('|Y(freq)|')
title('complete signal')
ticks=np.arange(0,Fs+1,F);
plt.xticks(ticks,ticks);



show()