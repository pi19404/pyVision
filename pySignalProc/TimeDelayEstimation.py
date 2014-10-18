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
from TDOA2 import *

class TimeDelayEstimator(object):
    def __init__(self):
        self.once=1;
        self.tvnoise=0;
        self.tspower=0;

    
    def run(self,signal,isignal,enoise,snoise,method=0,iteration=0):
        
        
        size1=signal.shape;
        ch=size1[1];
        length=size1[0];
        periods=size1[2];
        enoise=enoise*enoise;
        snoise=snoise*snoise;
        PS=np.mean(isignal*isignal);      
 


           
           
        if method==0:
            vsnoise=[-2*periods*snoise,2*periods*snoise]    
            ispower=(periods*periods*PS);
            spower=np.mean(signal*signal)*periods*periods
            OSNR=(periods*periods*PS)+(1*vsnoise);
            OSNR=OSNR/periods
        if method==1:
            vnoise=(periods*ch*ch*enoise)+(periods*ch*snoise);
            vsnoise=[-1*vnoise,1*vnoise];
            spower=np.mean(signal*signal)*periods*periods*ch*ch
            ispower=(periods*periods*ch*ch*PS);
            OSNR=(periods*periods*ch*ch*PS)+vsnoise
            OSNR=OSNR/(periods*periods)
            #print OSNR,vsnoise,(periods*periods*ch*ch*PS),PS
        
        s=np.mean(signal,axis=2);
        
        if method==0:
            iloop=ch
        if method==1:
            iloop=ch-1;
            
        SNR=np.zeros((iloop,length))
        for i in range(iloop):
            
            for k in range(length):
                if method==0:
                    su=s[:,i]+self.delay(isignal,k);
                if method==1:
                    su=s[:,1]+self.delay(s[:,0],k)
                    
                SNR[i,k]=np.mean(su*su);
                if(SNR[i,k]<OSNR[0]):
                    SNR[i,k]=0#(-SNR[i,k]);
         
        m1=np.argmax(SNR,axis=1)           
        if method==0:
            tdelay=abs(m1[0]-m1[1])
        if method==1:
            tdelay=abs(m1[0])+1;
        
        


           
           
        return tdelay   
        
    def delay(self,signal,N):
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
            
        d=signal[len(signal)-N-1:len(signal)];#numpy.zeros((1,N+1));    
        signal1=numpy.append(d,signal[0:len(signal)-N-1])
        return signal1;
        
        
        
class TimeDelaySimulator():
    def __init__(self,n_signals,n_periods,enoise,snoise,delay=5):
        self.nchannels=n_signals;
        self.periods=n_periods;
        self.enoise=enoise
        self.snoise=snoise
        self.tdelay=delay;
        self.seed=int(np.random.uniform(0,1)*10000);  
        self.once=1;
        random.seed(self.seed)
        self.tspower=0;
        self.tnpower=0;
    
    def run(self,isignal):
        l=len(isignal)
        
        signal=numpy.zeros((l,self.nchannels,self.periods));
        noise=numpy.zeros((l,self.nchannels,self.periods));    
        
        for k in range(self.periods):
            n=self.GenerateNoise(self.enoise,l)           
            #simulating environmental noise
            for i in range(self.nchannels):
                noise[:,i,k]=n;
            
            
        #over multiple sensors
        for k in range(self.nchannels):
            
            #dummpy call to change the random seed generator
            self.GenerateNoise(self.snoise,l,1)
            for i in range(self.periods):
                signal[:,k,i]=self.delay(isignal,(k+1)*self.tdelay)+noise[:,k,i]+self.GenerateNoise(self.snoise,l)    
        
        self.seed=int(np.random.uniform(0,1)*10000);   


        if self.once==1:
            self.once=0;
            PS=np.mean(isignal*isignal);
            PN=self.enoise*self.enoise+self.snoise*self.snoise;
            PNT=(self.nchannels*self.nchannels*self.periods*self.enoise*self.enoise)
            PNT=PNT+(self.periods*self.nchannels*self.snoise*self.snoise);
            PST=(self.periods*self.periods*self.nchannels*self.nchannels)*PS;
            print "signal strength",PS
            print "noise power ",PN
            self.SNRI=PS/PN
            print "Input SNR ",self.SNRI          
            print "estimated noise power ",PNT
            print "estimated signal power ",PST
            self.SNR0=PST/PNT
            print "estimated SNR ",self.SNR0
            print "estimated improvement ",self.SNR0/self.SNRI
                
        return signal

  
    def info(self,osignals,isignal,tdelay,iteration=0):
        
        signals=np.sum(osignals,axis=2);
        diff=np.zeros((signals.shape[0],signals.shape[1]));
        S=np.zeros((signals.shape[0],signals.shape[1]));

        for k in range(self.nchannels):
            diff[:,k]=signals[:,k]-self.delay(self.periods*isignal,(k+1)*tdelay)
            S[:,k]=signals[:,k]#+self.delay(self.periods*isignal,(k+1)*tdelay)
        
        diff=np.sum(diff,axis=1)
        #S=np.sum(S,axis=1);
        #S=S-(2*self.periods*isignal*self.nchannels)
        
        if iteration==0:
            self.tnpower=self.tnpower+np.mean(diff*diff,axis=0)
            self.tspower=self.tspower+np.mean(S*S)*self.nchannels*self.nchannels
        if iteration!=0:
            self.tnpower=self.tnpower+np.mean(diff*diff)
            self.tnpower=self.tnpower/iteration
            self.tspower=self.tspower+np.mean(S*S)*self.nchannels*self.nchannels
            self.tspower=self.tspower*1/iteration
            print "emprical signal power ",self.tspower;
            print "emprical noise power ",self.tnpower
            print "emprical SNR ",self.tspower/self.tnpower
        

    def delay(self,signal,N):
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
            return signal
            
        d=signal[len(signal)-N-1:len(signal)];#numpy.zeros((1,N+1));    
        signal1=numpy.append(d,signal[0:len(signal)-N-1])
        return signal1;
         
    def GenerateNoise(self,Variance,L,flag=0):
        """ function generats additive white gaussian noise 
        
        Parameters
        -----------
        L        : Integer
                   length of the signals
                 
        Variance : float
                   noise covariance
            
        Returns
        --------
        out : numpy-array
              noisy signal    
        
        """        
        if flag==1:
            random.seed(self.seed+100*Variance)
        else:
            random.seed(self.seed)
        noise = np.random.normal(0,Variance,L)  
        return noise;

        
        
if __name__ == "__main__":  
        iterations=100;
        Fs=5000;
        tdelay=10;
        varnoise1=0.4;
        varnoise2=0.4;
        
        #number of receivers
        loop1=4
        #number of periods of the signal
        loop2=1
                
        x2=4*rectangular(1000,100,200)     


                
        
        estimator=TimeDelayEstimator()
        
        simulator=TimeDelaySimulator(loop1,loop2,varnoise1,varnoise2)
        
        tdelay=np.zeros(iterations)
        for i in range(iterations):
            signals=simulator.run(x2);            
            tdelay[i]=estimator.run(signals,x2,varnoise1,varnoise2,1)
            simulator.info(signals,x2,tdelay[i],iterations*(i==iterations-1))
            
        print "mean delay ",np.mean(tdelay)
        print "std deviation ",np.std(tdelay)
        
        
        