#!/usr/bin/env sage -python

from pySignalProc.pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title
from scipy import interpolate
from tabulate import tabulate
from scipy import integrate

from pySignalProc.wavelet.pywtUtils import *

from pySignalProc.PieceWisePolynomial import *


#definitions for piecewise continuous functions
def f1(x):return 10
def f2(x):return 5*x
def f3(x):return 2*(x)**2
def f4(x):return (x)**3+(x)**2
def f5(x):return 20*((x)**5)
  

def sig(x,order):
    return (2*x+2)**order
    
#function to compute product of signal and wavelet basis    
def fsignal(x,order,wavelet):
    return sig(x,order)*wavelet(x)


w = [pywt.Wavelet('db1'),pywt.Wavelet('db2'),pywt.Wavelet('db4')]
fresult=[]
for k in range(len(w)):
    
    
    s1,w1,t=w[k].wavefun(level=20)
    wavelet=interpolate.interp1d(t, w1)
    scaling=interpolate.interp1d(t, s1)      
    #computing the mean squared errors
    result=[]
    result.append([w[k].name,"  "+str(w[k].vanishing_moments_psi)+" Vanishing moment"])
    result.append(["--------","--------","--------"])
    result.append(["order","wavelet integral","scaling integral"])
    result.append(["--------","--------","--------"])
    for i in range(0,5):
        ans,err=integrate.quad(fsignal, 0, t[len(t)-1],args=(i,scaling),limit=500)
        ans1,err1=integrate.quad(fsignal, 0, t[len(t)-1],args=(i,wavelet),limit=500)
        ans2,err2=integrate.quad(sig, 0, t[len(t)-1],args=(i),limit=500)
        result.append([i,ans1,ans2])

    #print tabulate(result)
    fresult.append(result)
    figure()
    plt.suptitle(w[k].name+" polynomial approximation")

#plot the projection coefficients
    result1=[]
    result1.append([w[k].name,"  "+ "MSE"])
    result1.append(["--------","--------","--------"])
    for i in range(0,5):
        x=np.linspace(0,2,128)    
        x=sig(x,i)
       
        coeff=pywt.wavedec(x,w[k],level=1)
        time,sig1,xx=plotWaveletProjection(x,x,w[k],0,0)
        time1,sig11,xx1=plotWaveletProjection(x,coeff[1],w[k],1,1)    
        
        
        plt.subplot(5,2,2*i+1)
        plt.plot(time,sig1)
        plt.grid()
        plt.fill_between(time,sig1,xx,facecolor='red')
        plt.subplot(5,2,2*i+2)
        plt.plot(time1,sig11)
        plt.grid()    
        d=coeff[1]
        coeff[1]=np.zeros(len(coeff[1]))
        signal=pywt.waverec(coeff,w[k])
        err=np.mean((signal-x)**2)
        result1.append([i,err])

    


    
    f = Piecewise([[(0,1),f5],[(1,2),f4],[(2,3),f3],[(3,4),f2],[(4,5),f1]])
    
    
    
    figure()
    plt.suptitle(w[k].name+" piecewise polynomial approximation")
    for i in range(0,5):
        x=np.linspace(0,5,128)
        
        x=f(x)
       
        coeff=pywt.wavedec(x,w[k],level=1,mode='per')
        #print len(coeff[0]),"ASDASD",len(x)
        
        #time,sig1,xx=plotProjection(x,coeff[0],w,1,0)
        time,sig1,xx=plotWaveletProjection(x,x,w[k],0,0)
        time1,sig11,xx1=plotWaveletProjection(x,coeff[1],w[k],1,1)
        
        
        plt.subplot(5,2,2*i+1)
        plt.plot(time,sig1)
        #plt.plot(time,sig1,'r')
        plt.fill_between(time,sig1,xx,facecolor='red')
        plt.subplot(5,2,2*i+2)
        plt.plot(time1,sig11)
            
        d=coeff[1]
        coeff[1]=np.zeros(len(coeff[1]))
        signal=pywt.waverec(coeff,w[k],mode='per')
        err=np.mean((signal-x)**2)
    #result1.append([i,err])

for i in fresult:
    print tabulate(i)    
show()