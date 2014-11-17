# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:21:57 2014

@author: pi19404
"""

from pyCommon import *
from numpy import linspace,cos,pi,ceil,floor,arange
from pylab import plot,show,axis
from numpy import hstack
from matplotlib.pyplot import figure

from tabulate import tabulate
from WaveSource import  *

def squared_sinc(f,points):
    x=numpy.sinc(f*points)
    return x*x

def sinc(f,points):
    x=numpy.sinc(f*points)
    return x
    

def linear_interpolation(x,t):
    
    interval=[] # piecewise domains
    apprx = []  # line on domains
    # build up points *evenly* inside of intervals
    tp = hstack([ linspace(t[i],t[i+1],20,False) for i in range(len(t)-1) ])
    
    # construct arguments for piecewise2
    for i in range(len(t)-1):
       interval.append( np.logical_and(t[i] <= tp,tp < t[i+1]))
       apprx.append( (x[i+1]-x[i])/(t[i+1]-t[i])*(tp[interval[-1]]-t[i]) + x[i])
       
    # piecewise linear approximation   
    x_hat = np.piecewise(tp,interval,apprx)     
    return x_hat,tp
    
    
def sinc_interpolation(x,t,fs1,fs,factor=None,tmax=None):
    num_coeffs=len(t) 
    tp = linspace(-factor,factor,2*factor*10*fs1)
    sm=0
    
    sm=np.zeros(len(tp))
    for k in range(num_coeffs): # since function is real, need both sides

            
        tau=-1.0+(1.0*(k+0.5))/fs1
        sig=x[k]*numpy.sinc((tp-tau)*fs1)
        #sig=Utils.delay(sig,fs,False)
        sm=sm+sig
        
    #sm=sm[len(sm)]
    #print sm
    #ddd
    #sm=numpy.sinc((tp-tp[0])*fs)
    #sm=sm[fs:len(tp)-fs]
    #tau=-1.0+(1.0*40/fs)
    #sm=numpy.sinc((tp)*fs)
    #sig=Utils.delay(sig,-fs,False)
    #sm=sig
    factor=factor-1
    sm1=sm[factor*10*fs1:len(tp)-factor*10*fs1]
    tp1=tp[factor*10*fs1:len(tp)-factor*10*fs1]
    #print tp1
    return sm1,tp1
    
#variables for display
display=True
display1=True
table=[]
table.append(['Sampling Rate','Mean squared error','Ratio'])
#support in time domain
tmin = -1;
tmax = 1;
fo=40
FS=[30]
#0.5
n=len(FS)
in1=1
k=1

fig = figure()
 
for Fs in FS:
    t = linspace(tmin, tmax, abs(tmax-tmin)*Fs);
    x=sinc(fo,t)
    
    #plot the frequeny spectrum of signal
    #if display:
    #Y,freq=Utils.plotSpectrum(x,Fs,'signal')
    
    
    # linear interpolation
    x_hat,tp=sinc_interpolation(x,t,Fs,Fs,4)
    #x_hat,tp=linear_interpolation(x,t)

    diff=sinc(fo,tp)-x_hat
    MSE=np.mean(diff*diff)
    
    print n,k,in1
    if display:
        ax1=fig.add_subplot(n,1,in1)
        ax1.plot(tp,x_hat)
        ax1.fill_between(tp,x_hat,sinc(fo,tp),facecolor='red')
        #ax1.set_xlabel('time',fontsize=18)
        #ax1.set_ylabel('Amplitude',fontsize=18)
        title='MSE',MSE
        ax1.set_title(title)
    
    
    #if in1==2:
        #k=k+1
        #in1=0
    
    #print MSE,table[in1-1][2]

        #print table[in1-1][1],MSE,table[in1-1][1]/MSE
    table.append([Fs,MSE,1e-4/MSE])
        
        
    in1=in1+1    
        
#print "Mean squared error at sampling rate ",Fs,"is ",MSE
#ax2 = ax1.twinx()
#sqe = ( x_hat - numpy.sinc(f*tp/2)*numpy.sinc(f*tp/2))**2
#ax2.plot(tp, sqe,'r')
#ax2.axis(xmin=-1,ymax= sqe.max() )
#ax2.set_ylabel('squared error', color='r',fontsize=18)
#ax1.set_title('Errors with Piecewise Linear Interpolant')

if display:
    show()
#show()    
print tabulate(table)
