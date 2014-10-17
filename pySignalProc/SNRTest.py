from numpy import *  
from scipy import fftpack, signal  
  


# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 22:50:07 2014

@author: pi19404
"""

import pylab
from PIL import Image
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import numpy

import scipy.signal as signal
import numpy, scipy, pylab, random  
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np;
import cmath
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange,ifft

import time

from scipy.signal import butter, lfilter,filtfilt
random.seed(1234)

def plotPSD(signal):
    
    #shape=f.shape[0];        
    r=np.fft.rfft(signal,axis=0)
    r=abs(r*r);
    r=np.mean(r,axis=1)/np.shape(r)[1]
    #r=fftpack.fftshift(r)
    r=20*np.log(abs(r))/np.log(10)
    plt.plot(range(len(r)),r)
    
    
        

def plotFrequency(b,a=1):
    
    """ the function plots the frequency and phase response """
    w,h = signal.freqz(b,a)
    h_dB = abs(h);#20 * np.log(abs(h))/np.log(10)
    subplot(211)
    plot(w/max(w),h_dB)
    #plt.ylim(-150, 5)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    subplot(212)
    h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
    plot(w/max(w),h_Phase)
    ylabel('Phase (radians)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)
    return w;
    
    
  
  
def nextpower2(x):  
    """ Computes the next power of two value of x. 
    E.g x=3 -> 4, x=5 -> 8 
    """  
    exponent = ceil(log(x)/log(2))  
    return 2**exponent  
    
    




once=0

def rem(n,a):
    return n%a;






    
   
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
        interval=end-start;
        x = np.zeros((1,N))
        x[:,start:start+interval]=1;
         
        x=x.flatten();    
        return x;

    

def triang(n=None):
  w=[];
  if (n==None):
    print "usage:  w = triang (n)"
    return 0;
  else :
    w = 1-abs(numpy.asarray(range(-(n-1),(n-1),2),dtype=float)/(n+rem(n,2)));

  return w;

def delay(signal,N):
    """ function introduces a delay of N samples 
    
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
    d=numpy.zeros((1,N+1));    
    signal1=numpy.append(d,signal)
    return signal1;


def GenerateNoise(Variance,L,flag=0):
    if flag==1:
        random.seed(1234+100*Variance)
    else:
        random.seed(1234)
    noise = np.random.normal(0,Variance,L)  
    return noise;
    
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
    
    noise=GenerateNoise(variance,len(s))
   
                     
    
    #print std(noise)
    #show()
    s=s+noise;                              
    return noise;

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
        if once==0:
            plt.figure(2)
            plotFrequency(b,a)
            once=1
        y = filtfilt(b, a, data)
        return y
    



if __name__ == "__main__":  
    
        carrier=None;
        Fs=5000;
        tdelay=10;
        varnoise1=0.4;
        varnoise2=0.4;
        
        #number of receivers
        loop1=2
        #number of periods of the signal
        loop2=4;

        
            
       
       
          
       
        #x=sinepulse(1000,100,200,carrier,Fs)
        x2=4*rectangular(1000,100,200)             
        
        PS=np.mean(x2*x2);
        PN=(varnoise2*varnoise2)+(varnoise1*varnoise1)
        print "Signal strength",PS
        print "Noise power",PN
        print "SNR",PS/PN
        SNR1=PS/PN;
         
       
        
        signal=numpy.zeros((1000,loop1,loop2));
        noise=numpy.zeros((1000,loop1,loop2));
        
        
        #over multiple periods of the signal
        for k in range(loop2):
            n=GenerateNoise(varnoise2,1000)           
            #simulating environmental noise
            for i in range(loop1):
                noise[:,i,k]=n;
            
            
        #over multiple sensors
        for k in range(loop1):
            
            #dummpy call to change the random seed generator
            GenerateNoise(varnoise1,1000,1)
            for i in range(loop2):
                signal[:,k,i]=x2+noise[:,k,i]+GenerateNoise(varnoise2,1000)
            #noise[:,i]=


        
        S=np.sum(signal,axis=1)              
        
        S=np.sum(S,axis=1);
        #print np.std(S,axis=0)
        PSA=np.mean(S*S);
        #
        print loop1*loop1*loop2*varnoise1*varnoise1 +loop1*loop2*varnoise2*varnoise2#loop1*(loop1*loop2+loop2)*PN/2
        
        #PSA=np.mean((S*S))
        print "Nosiy signal strength",PSA
        
                
        N=(S-loop1*loop2*x2);
        
        #NS=np.sum(N,axis=1)
        PNA=np.mean(N*N)
        print "noise power",PNA
        SNR2=PSA/PNA;
        print "SNR",SNR2
        print "SNR improvement",SNR2/SNR1
        
        #print "Mean SNR",(PSA/PN)
            ########
            
            
            

