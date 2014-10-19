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
    """ The class performs time delay estimation """
    def __init__(self):
        self.once=1;
        self.tvnoise=0;
        self.tspower=0;

    
    def run(self,signal,isignal,enoise,snoise,method=0):
        """ the main function that performs time delay estimation
        
        Parameters
        -----------
        signal : numpy array shape=(length,channel,periods)
                 input noisy signals for time delay estimation
                 
        isgignal : numpy array shape=(length,)
                   ideal signal
                   
        enoise,snoise : float 
                        environmental and sensor noise standard deviation
        
        method      : integer
                      0 - for pariwise delay sum with ideal signal
                      1 - pariwise delay sum with noisy signals
        
        Returns
        -------
        tdelay : numpy integer
                 estimated time delay
             
        
        """
        
        size1=signal.shape;
        ch=size1[1];
        length=size1[0];
        periods=size1[2];
        enoise=enoise*enoise;
        snoise=snoise*snoise;
        PS=np.mean(isignal*isignal);      
 


        if method==0:
            iloop=ch-1
        if method==1:
            iloop=ch-1;         
        
        #compute the SNR threshold
        if method==0:
            vnoise=(periods**enoise/4)+(periods*iloop*snoise/iloop);
            vsnoise=[-2*vnoise,2*vnoise]    
            ispower=(periods*periods*PS);
            spower=np.mean(signal*signal)*periods*periods
            OSNR=(periods*periods*4*PS)+(1*vsnoise);
            OSNR=OSNR/(periods*periods)
        if method==1:
            vnoise=(periods*enoise)+(periods*iloop*snoise);
            vsnoise=[-2*vnoise,1*vnoise];
            spower=np.mean(signal*signal)*periods*periods
            ispower=(periods*periods*iloop*iloop*PS);
            OSNR=(periods*periods*4*PS)+vsnoise
            OSNR=OSNR/(periods*periods)
            #print OSNR,vsnoise,(periods*periods*ch*ch*PS),PS
        
        
        #add the signals along periods
        s=np.mean(signal,axis=2);
        
  
        SNR=np.zeros((iloop,length))
        for i in range(iloop):
            #delay sum operation for various delays
            for k in range(length):
                if method==0:
                
                    su=s[:,0]+self.delay(s[:,i+1],k)
                if method==1:
                    su=s[:,0]+self.delay(s[:,i+1],k)
                
                #apply SNR Thresholding
                SNR[i,k]=np.mean(su*su);
                if(SNR[i,k]<OSNR[0]):
                   SNR[i,k]=0#(-SNR[i,k]);
        
        #find the index of maximum SNR
        m1=np.argmax(SNR,axis=1)          
   

        #average all the time delays        
        if method==0:
            tdelay=0;
            
            
            for i in range(iloop-1):
                #print m1,i
                tdelay=tdelay+abs(m1[0]-m1[i+1])/(i+1)
            tdelay=tdelay/(iloop-1)
        if method==1:
            
            tdelay=0;
            for i in range(iloop):
                tdelay=tdelay+((length-(abs(m1[i])+1)))/(i+1);
              
            
            tdelay=tdelay/iloop;
           
            

        #return the result  
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
    """ A Sumlator for generating noisy time delay signals """
    def __init__(self,n_signals,n_periods,enoise,snoise,tde_method,delay=5):
        """ Initialization function 
        Parameters
        ---------
        n_signals : Integer
                    number of receivers
                    
        n_periods   : integer
                      number of observation periods
                      
        enoise,snoise : float
                        standard deviation of environmental and sensor noise
                        
        tde_method : integer
                     0 - delay and sum with ideal signal
                     1 - delay and sum with noisy signals
        
        delay : integer
                Time delay to introduce
        
        """
        
        
        
        
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
        self.tde_method=tde_method;
    
    def run(self,isignal):
        """ main function that generates noisy,time delayed signals 
        
        Parameters 
        --------
        isignal : numpy array
                  ideal signal
        Returns
        -------
        out : numpy array shape=(length,channel,period)
            
        
        """
        l=len(isignal)
        #array for storing signal and noise
        signal=numpy.zeros((l,self.nchannels,self.periods));
        noise=numpy.zeros((l,self.nchannels,self.periods));    
        
        #generate environmental noise for each period and across all channels
        for k in range(self.periods):
            n=self.GenerateNoise(self.enoise,l)           
            #simulating environmental noise
            for i in range(self.nchannels):
                noise[:,i,k]=n;
            
            
        #for each channel add sensor noise
        for k in range(self.nchannels):
            
            #dummpy call to change the random seed 
            #so that we can generate uncorrelated sensor noise
            self.GenerateNoise(self.snoise,l,1)
            for i in range(self.periods):
                signal[:,k,i]=self.delay(isignal,(k+1)*self.tdelay)+noise[:,k,i]+self.GenerateNoise(self.snoise,l)    
        
        #change the seed everytime the function is called
        self.seed=int(np.random.uniform(0,1)*10000);   
            
        #flag to print the statistics 
        if self.once==1:
            self.once=0;
            PS=np.mean(isignal*isignal);
            PN=self.enoise*self.enoise+self.snoise*self.snoise;
           
            if self.tde_method==0:
                PNT=(self.periods*self.enoise*self.enoise)
                PNT=PNT+(self.periods*self.nchannels*self.snoise*self.snoise/self.nchannels)
                PST=(self.periods*self.periods)*PS*4              
                
            if self.tde_method==1:
                PNT=(self.periods*4*self.enoise*self.enoise)
                PNT=PNT+(self.periods*2*self.snoise*self.snoise)
                PST=(self.periods*self.periods)*PS*4
            
            print "signal strength",PS
            print "noise power ",PN
            self.SNRI=PS/PN
            print "Input SNR ",self.SNRI          
            print "estimated noise power ",PNT
            print "estimated signal power ",PST
            self.SNR0=PST/PNT
            print "estimated SNR ",self.SNR0
            print "estimated improvement ",self.SNR0/self.SNRI
        
        #returns the signal        
        return signal

  
    def info(self,osignals,isignal,tdelay,iteration=0):
        """ function prints emprical information 

        
        Parameters 
        --------
        isignal : numpy array
                  ideal signal
   
        osignals : numpy array shape=(length,channel,period)       
                   Time delayed noisy signals
                   
        tdelay : integer
                 estimated time delay
                 
        iteration : integer
                    N - last iteration
                    0 - other iteration
        
        Returns
        -------
        
        """
        
        
        if self.tde_method==0:
            channels=self.nchannels+1;
            osignals=osignals[:,1:osignals.shape[1],:]
        else:
            channels=self.nchannels
            
        signals=np.sum(osignals,axis=2);
        
      
        
        diff=np.zeros((signals.shape[0],signals.shape[1]));
        S=np.zeros((signals.shape[0],signals.shape[1]));

        #find the mean square error
        if self.tde_method==0:
            PS=np.zeros((self.nchannels));
            PN=np.zeros((self.nchannels));       
            for k in range(self.nchannels):
             
               S[:,k]=signals[:,k]+self.delay(self.periods*isignal,(k+1)*tdelay)
               diff[:,k]=S[:,k]-self.delay(2*self.periods*isignal,(k+1)*tdelay)
               PS[k]=np.mean(S[:,k]*S[:,k])
               PN[k]=np.var(diff[:,k])
    
 
        
            PS=np.mean(PS)
            PN=np.mean(PN)
            
        if self.tde_method==1:
            PS=np.zeros((self.nchannels-1));
            PN=np.zeros((self.nchannels-1));               
            for k in range(self.nchannels-1):               
               S[:,k]=signals[:,k+1]+self.delay(signals[:,0],(k+1)*tdelay)    
               diff[:,k]=S[:,k]-self.delay(2*self.periods*isignal,(k+2)*tdelay)
               PS[k]=np.mean(S[:,k]*S[:,k])
               PN[k]=np.var(diff[:,k])

            
          
            PS=np.mean(PS)
            PN=np.mean(PN)
            
              
        if iteration==0:
            self.tnpower=self.tnpower+PN
            self.tspower=self.tspower+PS
        if iteration!=0:
            self.tnpower=self.tnpower+PN
            self.tnpower=self.tnpower/(iteration)
            self.tspower=self.tspower+PS
            self.tspower=self.tspower*1/iteration
            print "emprical signal power ",self.tspower;
            print "emprical noise power ",self.tnpower
            ESNR0=self.tspower/self.tnpower;
            print "emprical SNR ",ESNR0
            print "emprical SNR improvement ",ESNR0/self.SNRI
            
        

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
        if Variance==0:
            return np.zeros(L);
        if flag==1:
            random.seed(self.seed+100*Variance)
        else:
            random.seed(self.seed)
        noise = np.random.normal(0,Variance,L)  
        return noise;

        
        
if __name__ == "__main__":  
        iterations=10;
        Fs=5000;

        #time delay
        tdelay1=5;
        #time delay estimation
        tde_method=0;
        
        #rectangular pulse signal        
        x2=4*rectangular(1000,100,200)     

        #mode to run various simulations
        mode=4
                
        
        estimator=TimeDelayEstimator()
        
        if mode==0:
            """ only environmental noise """
            #number of receivers
            loop1=2
            #number of periods of the signal
            loop2=1            
            #noise standard deviation
            varnoise1=0.2
            varnoise2=0.4;
        if mode==1:
            """ only sensor noise """
            #number of receivers
            loop1=2
            #number of periods of the signal
            loop2=4          
            #noise standard deviation
            varnoise1=0.4
            varnoise2=0.4; 
            
        if mode==2:
            """ environmental and sensor noise """
            #number of receivers
            loop1=8
            #number of periods of the signal
            loop2=4           
            #noise standard deviation
            varnoise1=0.4
            varnoise2=0.4;     
            
        if mode==3:
            """ time delay estimation using ideal signal """
            #number of receivers
            loop1=4
            #number of periods of the signal
            loop2=4         
            #noise standard deviation
            varnoise1=0.4
            varnoise2=0.4;   
            tde_method=0
            
        if mode==4:
            """ time delay estimation using noisy signals """
            
            #number of receivers
            loop1=4
            #number of periods of the signal
            loop2=4      
            #noise standard deviation
            varnoise1=0.4
            varnoise2=0.4;   
            tde_method=1;
        
        #creating simulator object
        simulator=TimeDelaySimulator(loop1,loop2,varnoise1,varnoise2,tde_method,tdelay1)
        #array for storing time delay
        tdelay=np.zeros(iterations)
        for i in range(iterations):
            #run the simulator
            signals=simulator.run(x2);  
            isignals=np.zeros((signals.shape[0],1,signals.shape[2]));
            #mode is zero ,add ideal signal in one of the channels
            if tde_method==0:            
                for k in range(loop2):
                    isignals[:,0,k]=x2;#
                    
                signals=np.append(isignals,signals,axis=1)

                    
            #perform time delay estimation
            tdelay[i]=estimator.run(signals,x2,varnoise1,varnoise2,tde_method)
            #display statistics on results
            simulator.info(signals,x2,tdelay[i],iterations*(i==iterations-1))
        
        #display final and mean
        print "mean delay ",np.mean(tdelay)
        print "std deviation ",np.std(tdelay)
        
        
        