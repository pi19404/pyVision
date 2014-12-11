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



def squared_sinc(f,points):
    x=numpy.sinc(f*points)
    return x*x
    
def sinc(f,points):
    x=numpy.sinc(f*points)
    return x

    
def pulsetran(f,l):
    x=np.ones(math.ceil(l*1.0/f),float)
    x=upsample(x,f)
    return x
    
def downsample(s, n, phase=0):
    """Decrease sampling rate by integer factor n with included offset phase.
    """
    return s[phase::n]


def upsample(s, n, phase=0):
    """Increase sampling rate by integer factor n  with included offset phase.
    """
    return numpy.roll(numpy.kron(s, numpy.r_[1, numpy.zeros(n-1)]), phase)    

def blit(t):
    """
    Return a periodic band-limited impulse train (Dirac comb) with
    period 2*pi (same phase as a cos)
 
    Examples
    --------
    >>> t = linspace(0, 1, num = 1000, endpoint = False)
    >>> f = 5.4321 # Hz
    >>> plot(blit(2 * pi * f * t))
 
    References
    ----------
    http://www.music.mcgill.ca/~gary/307/week5/bandlimited.html
    """
    t = np.asarray(t)
 
    if abs((t[-1]-t[-2]) - (t[1]-t[0])) > .0000001:
        raise ValueError("Sampling frequency must be constant")
 
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)
 
    # Get sampling frequency from timebase
    fs =  1 / (t[1] - t[0])
 
    # Sum all multiple sine waves up to the Nyquist frequency
    N = int(fs * pi) + 1
    for h in range(1, N):
        y += np.cos(h * t)
    y /= N
    return y
        
def butter_bandpass(lowcut, highcut, fs, order=5):
    """ function returns the bandpass butterworth filter coefficients 
    
    Parameters
    -------------    
    lowcut,highcut : integer
                     lower and higher cutoff freuencies in Hz
                     
    Fs : Integer
         Samping freuency in Hz

    order : Integer
            Order of butterworth filter                     
        
    Returns
    --------
    b,a - numpy-array
          filter coefficients 
          
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low,high], btype='bandpass')
    return b, a
    
def bandpass_filter(data, lowcut, highcut, fs, order,filter_type='butter_worth'):
    """ the function performs bandpass filtering 
    
    Parameters
    -------------
    data : numpy-array
           input signal
           
    lowcut,highcut : integer
                     lower and higher cutoff freuencies in Hz
                     
    Fs : Integer
         Samping freuency in Hz

    order : Integer
            Order of butterworth filter                     
        
    Returns
    --------
    out : numpy-array
          Filtered signal
    
    """
    global once
    if filter_type=='butter_worth':
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)            

        y = filtfilt(b, a, data)
        return y
    



     
     
def plotSpectrum(y,Fs,text=None,phase=False,time=True):
     """
     Plots a Single-Sided Amplitude Spectrum of y(t)
     """
     
     if time==True:
         index=2
     else:
         index=1
         
     n = len(y) # length of the signal
     #
     freq = np.fft.fftshift(np.fft.fftfreq(y.size,1.0)) # two sides frequency range
     #freq=freq[range(n/2)] # one side frequency range
     #print freq,y.size
     Y = np.fft.fftshift(numpy.fft.fft(y))/n # fft computing and normalization
     #np.fft.fftshift
     if phase==True:
         index=index+1
     else:
         index=index+1
         
     freq=freq*Fs
       
     seq=1
     if time==True:
         subplot(index,1,seq)
         plot(y)
         plot(range(len(y)),y,'ro')
         xlabel('Time')
         ylabel('Amplitude')
         if text!=None:
             plt.title(text)
             plt.grid()
         seq=seq+1
     
     subplot(index,1,seq)     
     plot(freq,(len(y)*abs(Y))**2,'r') # plotting the spectrum
     plt.grid()
     if Fs!=1:
         xlabel('Freq (Hz)')
         ylabel('|F(freq)|')
     else :
         xlabel('W (rad)')
         ylabel('|F(W)|')
     
     seq=seq+1
     
     if phase==True:
         subplot(index,1,seq)
         plot(freq,np.angle(Y)/math.pi,'r') # plotting the spectrum
         if Fs!=1:
             xlabel('Freq (Hz)')
             ylabel('P(freq)')
         else :
             xlabel('W (rad)')
             ylabel('P(W)')
         plt.grid()
     
     #plotSpectrum(y,Fs)
     #show()    
     return Y,freq
        
    
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

def delay(signal,N,circular=True):
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
    if circular:
        signal1=numpy.append(d,signal[0:len(signal)-N])
    else:
        signal1=numpy.append(d,np.zeros(len(signal)-N))
    return signal1;
    
def sinepulse(N,start,end,f ,Fs=1000,tau=0):
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
    
    
    
    
    
    
    
    
    
    