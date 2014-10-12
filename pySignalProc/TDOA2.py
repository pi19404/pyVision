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
import GCC
import time

from scipy.signal import butter, lfilter,filtfilt

once=0

def rem(n,a):
    return n%a;



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
    signal=numpy.append(d,signal)
    return signal;
    
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
        varnoise=0.3;
        loop=1000
        order=2;
        filter_type='butter_worth'
        #filter_type='FIR'
        carrier=1000;
        
        mode=2;
        if mode==0:
            x=sinepulse(1000,100,200,carrier,Fs)
            x2=rectangular(1000,100,200)            
            plt.figure(2)         
            plotFrequency(x)
            
            
            s = bandpass_filter(x, carrier-500, carrier+500, Fs,order,filter_type)
            filtered_signal=s;
            #filtered_signal[0:10]=0.0
            
            
            plt.figure(1)
            subplot(2,1,1) 
            plt.plot(range(len(x)),x)
            xlabel('Time')
            ylabel('Amplitude')
        
            #if carrier!=None:
            subplot(2,1,2)         
            plt.plot(range(len(filtered_signal)),filtered_signal)
            title('band pass filtered')
            xlabel('Time')
            ylabel('Amplitude')            
       
        if mode==1:     
            x=sinepulse(1000,100,200,carrier,Fs)
            x2=rectangular(1000,100,200)             
            
        if mode==2:     
            x=sinepulse(1000,100,200,carrier,Fs)
            x2=rectangular(1000,100,200)             

        
        #introduce delay
        dx=delay(x,tdelay);
        result=numpy.zeros((1,loop));
        for i in range(loop):
            s=dx;

            #add noise            
            if varnoise!=0:
                s=addNoise(s,varnoise)

            #normalize signals                
            s=s/np.linalg.norm(s);
            x1=x2/np.linalg.norm(x2);
            if carrier!=None:
                unfiltered_signal=s;
                if mode==1:
                    s = bandpass_filter(s, carrier-500, carrier+500, Fs,order,filter_type)
                if mode==2:
                    s = bandpass_filter(s, carrier-750, carrier+750, Fs,order,filter_type)

                filtered_signal=s;
            else:
                unfiltered_signal=s;
             
            #perform envelope detection
            s=abs(scipy.signal.hilbert(s))
            if mode==1 or mode==2:
                r=numpy.correlate(s,x1,mode="full")
      


   
        #plot the data and results
        c=np.sum(result==tdelay)
        ic=np.sum(result!=tdelay)
 
        print " *********** Information ************ "
        print 'time delay : ',tdelay
        print 'Noise :',varnoise
        if varnoise!=0:
            print "SNR",10*np.log(np.mean(abs(x*x))/(varnoise*varnoise))/np.log(10);
        print "Correct ",str(c)," Incorrect ",str(ic)
        print "mean ",np.mean(result),"Std ",np.std(result)
        #print result
        plt.figure(1)
        subplot(2,2,1) 
        plt.plot(range(len(x)),x)
        xlabel('Time')
        ylabel('Amplitude')

        
        subplot(2,2,2) 
        plt.plot(range(len(unfiltered_signal)),unfiltered_signal)
        xlabel('Time')
        ylabel('Amplitude')

        
        
        subplot(2,2,3) 
        plt.plot(range(len(r)),r)
        xlabel('Time')
        ylabel('Amplitude')
        
        if carrier!=None:
            subplot(2,2,4)         
            plt.plot(range(len(filtered_signal)),filtered_signal)
            title('band pass filtered')
            
            
        show()

