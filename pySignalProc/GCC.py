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
    
    
def GCC(x,y,filt="unfiltered",fftshift=1, b=None, a=None):  
    """ Generalized Cross-Correlation of _real_ signals x and y with 
    specified pre-whitening filter. 
     
    The GCC is computed with a pre-whitening filter onto the 
    cross-power spectrum in order to weight the magnitude value 
    against its SNR. The weighted CPS is used to obtain the 
    cross-correlation in the time domain with an inverse FFT. 
    The result is _not_ normalized. 
     
    See "The Generalized Correlation Method for Estimation of Time Delay" 
    by Charles Knapp and Clifford Carter, programmed with looking at the 
    matlab GCC implementation by Davide Renzi. 
     
    x, y      input signals on which gcc is calculated 
    filt      the pre-whitening filter type (explanation see below) 
    fftshift  if not zero the final ifft will be shifted 
    returns   the gcc in time-domain 
     
    descibtion of pre-whitening filters: 
     
    'unfiltered': 
    performs simply a crosscorrelation 
     
    'roth': 
    this processor suppress frequency regions where the noise is 
    large than signals. 
     
    'scot': Smoothed Coherence Transform (SCOT) 
    this processor exhibits the same spreading as the Roth processor. 
     
    'phat': Phase Transform (PHAT) 
    ad hoc technique devoleped to assign a specified weight according 
    to the SNR. 
     
    'cps-m': SCOT filter modified 
        this processor computes the Cross Power Spectrum Density and 
        apply the SCOT filter with a power at the denominator to avoid 
        ambient reverberations that causes false peak detection. 
     
    'ht': Hannah and Thomson filter (HT) 
        HT processor computes a PHAT transform weighting the phase 
        according to the strength of the coherence. 
     
    'prefilter': general IIR prefilter 
    with b and a a transfer function of an IIR filter can be given, which 
    will be used as a prefilter (can be a "non-whitening" filter) 
     
    2007, Georg Holzmann 
    """  
    L = max( len(x), len(y) )  
    fftsize = int(2**ceil(log(L)/log(2))) # next power2  
    X = np.fft.rfft(x, fftsize)  
    Y = np.fft.rfft(y, fftsize)  
      
    # calc crosscorrelation  
    Gxy = X.conj() * Y  
      
    # calc the filters  
      
    if( filt == "unfiltered" ):  
        Rxy = Gxy  
          
    elif( filt == "roth" ):  
        Gxx = X.conj() * X  
        W = ones(Gxx.shape,Gxx.dtype)  
        W[Gxx!=0] = 1. / Gxx[Gxx!=0]  
        Rxy = Gxy * W  
          
    elif( filt == 'scot' ):  
        Gxx = X.conj() * X  
        Gyy = Y.conj() * Y  
        W = ones(Gxx.shape,Gxx.dtype)  
        tmp = sqrt(Gxx * Gyy)  
        W[tmp!=0] = 1. / tmp[tmp!=0]  
        Rxy = Gxy * W  
      
    elif( filt == 'phat' ):  
        W = ones(Gxy.shape,Gxy.dtype)  
        
        Gxx = X.conj() * X 
        tmp = abs(Gxy)  
        
        
        #W[tmp>10] = 1. / tmp[tmp>10]  
        #W=abs(Gxy);
        W[tmp>=10]=1./tmp[tmp>=10];        
        #plotFrequency(np.fft.irfft(W))
        #show()
        Rxy = Gxy * W  
      
    elif( filt == 'cps-m' ):  
        Gxx = X.conj() * X  
        Gyy = Y.conj() * Y  
        W = ones(Gxx.shape,Gxx.dtype)  
        factor = 0.75 # common value between .5 and 1  
        tmp = (Gxx * Gyy)**factor  
        W[tmp!=0] = 1. / tmp[tmp!=0]  
        Rxy = Gxy * W  
      
    elif( filt == 'ht' ):  
        Gxx = X.conj() * X  
        Gyy = Y.conj() * Y  
        W = ones(Gxy.shape,Gxy.dtype)  
        gamma = W.copy()  
        # coherence function evaluated along the frame  
        tmp = sqrt(Gxx * Gyy)  
        gamma[tmp!=0] = abs(Gxy[tmp!=0]) / tmp[tmp!=0]  
        # HT filter  
        tmp = abs(Gxy) * (1-gamma)  
        W[tmp!=0] = gamma[tmp!=0] / tmp[tmp!=0]  
        Rxy = Gxy * W  
      
    elif( filt == 'prefilter' ):  
        # calc frequency response of b,a filter  
        impulse = zeros(fftsize)  
        impulse[0] = 1  
        h = signal.lfilter(b,a,impulse)  
        H = fft.rfft(h, fftsize)  
        Rxy = H * Gxy  # hm ... conj da irgendwo ?  
          
    else:  
        raise ValueError, "wrong pre-whitening filter !"  
      
    # inverse transform with optional fftshift  
    if( fftshift!=0 ):  
        gcc = fftpack.fftshift( np.fft.irfft( Rxy ) )  
    else:  
        gcc = np.fft.irfft( Rxy )  
    return gcc  
  
  
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
    signal1=numpy.append(d,signal)
    return signal1;


def GenerateNoise(Variance,L):
     N = 1024; 
     MU=0*numpy.ones((1,N));
     Cxx=(Variance)*np.diag(np.ones((1,N)).reshape(N,))
     ##Cxx=(Variance)*np.diag(np.ones((N,1)));
     
     R = np.linalg.cholesky(Cxx).T;
     NN=np.random.normal(0,Variance,(L,N));
     
     z =  np.dot(NN,R);     
     plotPSD(z)
     #plotFrequency(z)
     show();
    
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
    
    #GenerateNoise(variance,len(s))
    noise = np.random.normal(0,variance,len(s))                   
    
    #print std(noise)
    #show()
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
        varnoise=0.5;
        loop=1000
        order=2;
        filter_type='butter_worth'
        #filter_type='FIR'
        carrier=1000;
        
        mode=3;
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

        if mode==3:     
            x=sinepulse(1000,100,200,carrier,Fs)
            x2=rectangular(1000,100,200)             
            xa1=sinepulse(1000,100,200,carrier,Fs)
            x12=rectangular(1000,100,200)             
            
        
        #introduce delay
        dx=delay(x,tdelay);
        dx1=delay(xa1,tdelay);
        result=numpy.zeros((1,loop));
        for i in range(loop):
            s=dx;
            s1=dx1;

            #add noise            
            if varnoise!=0:
                s=addNoise(s,varnoise)
                s1=addNoise(s1,varnoise)

            #normalize signals                
            s=s/np.linalg.norm(s);
            s1=s1/np.linalg.norm(s1);
            x1=x2/np.linalg.norm(x2);
            x11=x12/np.linalg.norm(x12);
            if carrier!=None:
                unfiltered_signal=s;
                if mode==1:
                    s = bandpass_filter(s, carrier-500, carrier+500, Fs,order,filter_type)
                if mode==2 :
                    s = bandpass_filter(s, carrier-500, carrier+500, Fs,order,filter_type)
                        
                if mode==3:
                    s = bandpass_filter(s, carrier-500, carrier+500, Fs,order,filter_type)
                    s1 = bandpass_filter(s1, carrier-500, carrier+500, Fs,order,filter_type)
                    s=0.5*(s+s1)
                    
                filtered_signal=s;
            else:
                unfiltered_signal=s;
             
             
            ########
            
            
            
          
            #perform envelope detection
            s=abs(scipy.signal.hilbert(s))
            if mode==1 or mode==2 :
                r=numpy.correlate(s,x1,mode="full")

                diff=abs(len(r)-len(x1))/2
                r=r[diff:len(r)-diff]                
                #print len(x1),len(r)
                #ddd

                arg=np.argmax(r)
                result[0,i]=abs(arg-len(x)/2)   
                
            if mode==3:
                r=GCC(x1,s,"unfiltered",1);      
              
                diff=abs(len(r)-len(x1))/2
                r=r[diff:len(r)-diff]                

                arg=np.argmax(r)
                result[0,i]=abs(arg-len(x)/2)   
   
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

