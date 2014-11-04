#PKG = 'simulator1'
#import roslib; 
#roslib.load_manifest(PKG)
#from simulator1.msg import Transmitter as tmsg

#import rospy
#from std_msgs.msg import String
#import rospy
#from std_msgs.msg import String

from math import atan2, degrees, pi
import numpy as np;

import pylab
from PIL import Image
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import numpy


import numpy, scipy, pylab, random  
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np;
import cmath
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange,ifft

import time
import math



from scipy.signal import butter, lfilter,filtfilt

from numpy.linalg import *
import fractions



cexp=numpy.vectorize(cmath.exp);
acos=numpy.vectorize(math.acos);

        

        
    
def distance(source,destination):
    diff=source-destination;
    diff=np.array(diff)
   
    dist=np.sum(diff*diff);
 
    dist=np.sqrt(dist)
    return dist;
    
    
    

def phase_sift(signal,phase):
    Fsignal=np.fft.fft(signal)
    freq = np.fft.fftfreq(Fsignal.size,1.0)
    Fsignal=Fsignal*np.exp(-1j * np.pi *freq)  
    signal=np.real(np.fft.ifft(Fsignal))
    return signal

def angle(source,dest):
    #print source,dest
    source=source.flatten();
    dest=dest.flatten();
    dx = dest[0] - source[0]
    dy = dest[1] - source[1]
    rads = atan2(dy,dx)
    rads %= 2*pi
    degs = degrees(rads)     
    return degs-180


def convolution(signal,h):
    """ function that performs linear convolution """
    output=scipy.convolve(signal,h,"same")    
    return output
    
    
def fdelay(signal,N,mode="linear"):
    """ function introduces a fractional delay of N samples 
    
    Parameters
    -----------
    signal : numpy-array,
             The input signal
             
    N      : factional
             delay
        
    Returns
    --------
    out : numpy-array
          delayed signal
    
    """    
 
    if mode=="linear":      
        f,i=math.modf(N)
        #perform integral delay
        signal=delay(signal,i)           
        #perform linear interpolation for fractional delay    
        output=convolution(signal,[f,1-f])
    if mode=="upsample":
        N=math.ceil(N*100)/100
        #get rational approximation
        result=fractions.Fraction(N).limit_denominator(20)
        num=result.numerator;
        den=result.denominator
        #upsample the signal and interpolate

        out1=scipy.signal.resample(signal,den*len(signal))
        #delay the signal
        out1=delay(out1,int(num))        
        #downsample the signal

        out1=scipy.signal.resample(out1,len(signal))
        
        output=out1
         
    return output

def delay(signal,N):
    """ function introduces a circular delay of N samples 
    
    Parameters
    -----------
    signal : numpy-array,
             The input signal
             
    N      : integer
             delay
        
    Returns
    --------
    out : numpy-array
          delayed signal
    
    """
    if N==0:
        return signal;
        
    if N >= len(signal):
        N=N%len(signal)
    
    if N <0:
        N=N%len(signal)
    
   
    d=signal[len(signal)-N:len(signal)];#numpy.zeros((1,N+1));    
    signal1=numpy.append(d,signal[0:len(signal)-N])
    return signal1;
    
def sinepulse(N,start,end,f,Fs=1000,tau=0):
        """ function generates modulated rectangular pulse

        Parameters
        ---------
        N : integer
            length of signal
            
        start,end: integer
                starting and ending index of pulse
                
        f   : integer
              modulated carrier freuency
              
        Fs  : integer
              Sampling freuency

        tau : integer
              carrier phase delay in samples.
              
        Returns
        --------
        out : numpy-array
              modulated rectangular  signal               
          
        """
        x = np.zeros((1,N))
        t=np.asarray(range(0,end-start));
        x[:,start:end]=np.cos(2*np.pi*f*(t+tau)/Fs);
        
        return x.flatten();

def rectangular(N,start,end):
        """ fuction generates a rectangular pulse 
        
        Parameters
        ---------
        N : integer
            length of signal
            
        start,end: integer
                starting and ending index of pulse
 
    Returns
    --------
    out : numpy-array
          rectangular pulse signal
          
        """
        x = np.zeros((1,N))
        x[:,start:end]=1;
        x=x.flatten();    
        return x;

    



    
def addNoise(s,variance):
    """ function add additive white gaussian noise to the input signal 
    
    Parameters
    -----------
    s : numpy-array,
             The input signal
             
    N      : float
             noise covariance
        
    Returns
    --------
    out : numpy-array
          noisy signal    
    
    """
    noise = np.random.normal(0,variance,len(s))                    
    s=s+noise;                              
    return s;


   
#    res=v4.angle(v4,v1)
#    print res*180/math.pi
# 
#    res=v5.angle(v5,v1)
#    print res*180/math.pi
    
    
    
    
    
    
    
    
    
    