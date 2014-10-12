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


if __name__ == "__main__":  
    
        carrier=None;
        Fs=1000;
        
        mode=6
        if mode==0:
            x = triang(20);
            x2=x;
        if mode==1:
            x=rectangular(20,8,14);
            x2=x;
        if mode==2:
            x=rectangular(20,8,14)+rectangular(20,2,5)
            x2=x;            
        if mode==3:
            x=rectangular(1000,100,150)
            x2=x;
        if mode==4:
            x=sinepulse(1000,100,200,100)
            x2=x;
        if mode==5:
            x=sinepulse(1000,100,200,100)
            x2=sinepulse(1000,100,200,50,Fs,1)
        if mode==6:
            x=sinepulse(1000,100,200,100,Fs)
            x2=rectangular(1000,100,200)
        if mode==7:
            x=sinepulse(1000,100,200,100,Fs)
            x2=rectangular(1000,100,200)
            
            
        tdelay=10;
        varnoise=0.001;
        loop=1000

        
        #delay the signal
        dx=delay(x,tdelay);
        result=numpy.zeros((1,loop));
        for i in range(loop):
            s=dx;
            #add noise
            s=addNoise(s,varnoise);
            
            #normalize the signals
            s=s/np.linalg.norm(s);
            # x2 contains rectangular passband signal in case of envelope detection
            x1=x2/np.linalg.norm(x2);
            
            unfiltered_signal=s;

            if mode==5 or mode ==6 or mode==7:                
                s=abs(scipy.signal.hilbert(s))
            
            r=numpy.correlate(s,x1,mode="full")
            arg=np.argmax(r)
            result[0,i]=abs(arg-len(x))     
            
            #print 
        


        
        #plot the results
        c=np.sum(result==tdelay)
        ic=np.sum(result!=tdelay)
        #10*np.log(np.sum(np.abs(x*x))/
        print " *********** Information ************ "
        print 'time delay : ',tdelay
        print 'Noise :',varnoise
        print "SNR",10*np.log(np.mean(abs(x*x))/(varnoise*varnoise))/np.log(10);
        print "Correct ",str(c)," Incorrect ",str(ic)
        print "mean ",np.mean(result),"Std ",np.std(result)
        print result
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
        


#        subplot(2,2,4)
#        print s.shape
#        p=np.zeros((1,s.shape[0]));
#        p[0,:]=s;
#        #p[1,:]=noise;
#        n, bins, patches =plt.hist(p.T*100,50,histtype='bar',normed=1);#color=['crimson', 'burlywood'])
#        #plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')        
        #plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)        

        
        show()        

