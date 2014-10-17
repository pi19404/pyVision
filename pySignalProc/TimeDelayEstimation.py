# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 22:03:47 2014

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


class TimeDelayEstimation(object):
    
    
    
    def delay(self,signal,N):
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
                