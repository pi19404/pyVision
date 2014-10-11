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
from scipy import fft, arange,ifft
import time

cexp=numpy.vectorize(cmath.exp);
cabs=numpy.vectorize(abs);

import cmath

    
    
def FourierSeries(input,N=None):
    """ computes the fourier series coefficients of input signal
    
    Parameters
    -----------
    input : numpy array
            input discrete time signal
            
    Returns
    --------
    out : complex,list
          fourier series coefficients
    """
    
    N=len(input);

    w=2*cmath.pi/N;
    input=input[0:N];
    n=numpy.arange(0,N);    
    r=cexp(-1j*w*n);

    output = [complex(0)] * N    
    for k in range(N):        
        r=input*cexp(-1j*w*n*k)          
        output[k]=np.sum(r);
        
   
    return output;
    
   
def IFourierSeries(input):
    """ function reconstructs the signal from fourier series coefficients
    
    Parameters
    ---------
    input : cmath list
            fourier series coefficients    
    
    Returns
    -----------
    out : numpy arrary
          reconstructed signal
    
    """
    N=len(input);
    w=2*cmath.pi/N;
    k=numpy.arange(0,N);    
    output = [complex(0)] * N   
    for n in range(N):  
        r=input*cexp(-1j*w*n*k);
        output[n]=np.mean(r);

    print output.__class__    
    return output;
 

def FourierRect(N):     
        """ the function generates rectangular aperiodic pulse and computer the DFT coefficients
        
        Parameters
        ----------
        N : period of aperiodic signal
             
        """
        x = np.zeros((1,N))
        x[:,0:30]=1;
        x=x.flatten();
    
        
        #compute the DFT coefficients
        r1=FourierSeries(x)
        #magnitude of DFT coefficients
        a1=cabs(r1)

        #plot the time domain signal
        subplot(2,1,1)
        plt.plot(range(0,len(x)),x)
        xlabel('Time')
        ylabel('Amplitude')
        title('time doman')
        plt.ylim(-2,2);
        
        #plot the DFT coefficients
        L=len(a1);
        fr=np.arange(0,L);
        subplot(2,1,2)
        plt.stem(fr,a1,'r') # plotting the spectrum
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
        title('complete signal')
        ticks=np.arange(0,L+1,25);
        plt.xticks(ticks,ticks);     
        show()         
   

def FourierSinusoids(F,w,Fs,synthesis=None):    
    """ the function generates a discrete time sinusoid,computes
	the fourier series coefficients and plots the time and frequency
	data 
 
     Paraneters
     ----------
     F : numpy array
         sinuoidal frequency components
         
     w  : numpy array
          the weights associated with freuency components
         
     Fs : Integer
          sampling frequency
          
     synthesis : int
                 if 1 ,reconstructs signal and plots the original and reconstructed
                 signal
 
    """
    if synthesis==None:
        synthesis=0;
        
    Ts=1.0/Fs;   
    xs=numpy.arange(0,1,Ts) 
    
    signal=numpy.zeros(np.shape(xs));
    for i in range(len(F)):
        omega=2*np.pi*F[i];
        signal = signal+ w[i]*numpy.cos(-omega*xs);
    #plot the time domain signal    
    subplot(2,1,1)
    plt.plot(range(0,len(signal)),signal)
    xlabel('Time')
    ylabel('Amplitude')
    title('time doman')
   
    #compute the fourier series coefficient
    r1=FourierSeries(signal)
    a1=cabs(r1)
    
    if synthesis==0:
        #plot the freuency domain signal
        L=len(a1);
        fr=np.arange(0,L);
        subplot(2,1,2)
        plt.stem(fr,a1,'r') # plotting the spectrum
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
        title('complete signal')
        ticks=np.arange(0,L+1,25);
        plt.xticks(ticks,ticks);     
        show() 
        
    if synthesis==1:
        rsignal=IFourierSeries(r1);
        print np.allclose(rsignal, signal)    
        subplot(2,1,2) 
        plt.stem(xs,signal)
        xlabel('Time')
        ylabel('Amplitude')
        title('reconstructed signal')
        show() 
        
# Greatest common divisor of more than 2 numbers.  Am I terrible for doing it this way?
 
from fractions import gcd
    
def lcm(numbers):
    return reduce(lambda x, y: (x*y)/gcd(x,y), numbers, 1)
      

if __name__ == "__main__":     

    mode =7
    
    if mode==0:
        F=[10];
        F=np.array(F);
        w=numpy.ones(F.shape);
        #plot the time domain signal and fourier series component
        FourierSinusoids(F,w,150);

    if mode==1:
        F=[10,20];
        F=np.array(F);
        w=numpy.ones(F.shape);
        FourierSinusoids(F,w,150);        
    
        
    if mode==2:
        F=range(1,10);
        F=np.array(F);
        w=numpy.ones(F.shape);
        FourierSinusoids(F,w,150); 

    if mode==3:
        F=range(1,10);
        F=np.array(F);
        w=numpy.ones(F.shape);
        FourierSinusoids(F,w,150,1);         
        
    if mode==4:
       FourierRect(150);

    if mode==5:
       FourierRect(150*3);
       
    if mode==6:
        Fs=150;
        F=range(1,10);
        F=np.array(F);        
        w=numpy.ones(F.shape);

        
        Ts=1.0/Fs;   
        xs=numpy.arange(0,1,Ts) 
    
        signal=numpy.zeros(np.shape(xs));
        for i in range(len(F)):
            omega=2*np.pi*F[i];
            signal = signal+ w[i]*numpy.cos(omega*xs);        
            
            
        start_time = time.time()
        FourierSeries(signal)
        end_time = time.time()
        print("Elapsed time naive algo  %g seconds" % (end_time - start_time)) 

        start_time = time.time()
        fft(signal)
        end_time = time.time()
        print("Elapsed time of fft algo  %g seconds" % (end_time - start_time)) 
        
        
    if mode == 7:
        x=np.array([1,1,1,1,1]);
        h=np.array([1,2,3,4,0]);
        r=np.convolve(x,h)
        print 'Linear convolution'
        print r
        f1=fft(x,5);
        f2=fft(h,5);
        s=ifft(f1*f2);
        print 'Circular convolution'
        print abs(s)
        f1=fft(x,9);
        f2=fft(h,9);
        s=ifft(f1*f2);
        print 'zero padded Circular convolution'
        print r
        
    #r1=FourierSeries(signal)
    
    #print r
    #a=cabs(r)
        
        #print a

        #mode=0
    
    
        #subplot(2,1,2)
        #L=len(x);
        
        
        

        
#    
#
#
#        
#    if mode==5:
#        from scipy import signal
#        import matplotlib.pyplot as plt
#        subplot(2,1,2)
#        x = np.zeros((1,N))
#        x[:,0:30]=1;
#        x=np.flatten(x);
#        
        
    
    