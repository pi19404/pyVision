# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 12:27:18 2014

@author: pi19404
"""

from pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title
from scipy import interpolate
from tabulate import tabulate
from scipy import integrate




w = pywt.Wavelet('db4')
print w
s1,w1,t=w.wavefun(level=20)
print t[len(t)-1],"XX"
plt.plot(t,w1,'r')
#plt.plot(w.dec_lo,'ro')
plt.plot(t,s1,'g')
#plt.plot(w.dec_hi,'go')
wavelet=interpolate.interp1d(t, w1)
scaling=interpolate.interp1d(t, s1)

def sig(x,order):
    return (2*x+2)**order
    
    
def f(x,order):
    return sig(x,order)*wavelet(x)

def f1(x,order):
    return sig(x,order)*scaling(x)
    


def plotProjection(x,coeff,w,level=1,mode=0):
    s1,w1,t2=w.wavefun(level=5)
    time=[]
    sig=[]
    s1=np.array(s1,float)
    
    length=len(w.dec_lo)
    
    l=2**level
    end1=math.floor(t2[len(t2)-1])
    d=abs(float(len(coeff))*float(l)-len(x))
    d=int(d)
    
    for i in range(len(coeff)-1):     
       
       #t2[len(t2)-1]
       t=np.linspace(l*i,l*i+l*end1,l*end1*(128));
       
       t1=np.linspace(0,end1,end1*(128)*l);
       
       if mode==0:
           val=coeff[i]*scaling(t1)
       else:
           val=coeff[i]*wavelet(t1)
           

       ratio=end1
       #print end1,t2[len(t2)-1]
       inc=len(t)/ratio
       #for k in range(ratio):
       if i==0:
               sig=np.append(sig,val)
               time=np.append(time,t)
               
       else:          
               v1=val[0:len(t)-inc]
               #print inc,len(t),i,i*inc,"AA",len(coeff),length,len(x)
               if  inc < len(t):
                   v2=val[len(t)-inc:len(t)]
                   sig[i*inc:i*inc+len(t)]=sig[i*inc:i*inc+len(t)]+v1
                   sig=np.append(sig,v2)
                   time=np.append(time,np.linspace(l*i+l*end1-l,l*i+l*end1,inc))       
               else:
                   sig=np.append(sig,val)
                   time=np.append(time,t)      
    
       
    sig=np.array(sig).flatten()
    time=np.array(time).flatten().ravel()
    #print sig.shape,time.shape
    sig=sig/(math.sqrt(2)**level)
    #plt.plot(time,sig)
    #plt.grid()

    #plt.stem(range(d,d+len(x)),x,'r')
    
    #r=range(d**(level-1)/8,len(time)-d**(level-1)/9)
   
    #x=np.append(np.zeros(d),np.repeat(x,len(sig)/len(x)))
    
    x=np.repeat(x,len(sig)/len(x))
    #if mode==0:
    #    
  
    return time,sig,x

    
result=[]
result.append([w.name,"  "+str(w.vanishing_moments_psi)+" Vanishing moment"])
#result.append(["--------","--------","--------"])
#result.append(["order","wavelet integral","scaling integral"])
#result.append(["--------","--------","--------"])
#for i in range(0,5):
#    ans,err=integrate.quad(f, 0, t[len(t)-1],args=(i),limit=500)
#    ans1,err1=integrate.quad(f1, 0, t[len(t)-1],args=(i),limit=500)
#    ans2,err2=integrate.quad(signal, 0, t[len(t)-1],args=(i),limit=500)
#    result.append([i,ans,ans2])
#
print tabulate(result)
figure()
result1=[]
result1.append([w.name,"  "+ "MSE"])
result1.append(["--------","--------","--------"])
for i in range(0,5):
    x=np.linspace(0,2,128)
    
    x=sig(x,i)
   
    coeff=pywt.wavedec(x,w,level=1)
    #print len(coeff[0]),"ASDASD",len(x)
    
    #time,sig1,xx=plotProjection(x,coeff[0],w,1,0)
    time,sig1,xx=plotProjection(x,x,w,0,0)
    time1,sig11,xx1=plotProjection(x,coeff[1],w,1,1)
    
    
    plt.subplot(5,2,2*i+1)
    plt.plot(time,sig1)
    plt.fill_between(time,sig1,xx,facecolor='red')
    plt.subplot(5,2,2*i+2)
    plt.plot(time1,sig11)
        
    d=coeff[1]
    coeff[1]=np.zeros(len(coeff[1]))
    signal=pywt.waverec(coeff,w)
    err=np.mean((signal-x)**2)
    result1.append([i,err])




#
#print tabulate(result1)
show()