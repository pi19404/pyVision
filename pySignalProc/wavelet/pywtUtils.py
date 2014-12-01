from pySignalProc.pyCommon import *
import pywt

from scipy import interpolate

from scipy import integrate



def plotWaveletProjection(x,coeff,w,level=1,mode=0):
    """ function plots the projection of signal on the scaling and wavelet functions subspace
    
    Parameters
    -----------
    x           : numpy-array,
                  The input signal
             
    coeff      : numpy=-array
                 The wavelet or scaling coefficient
                 
    w           : pywt.Wavelet
                  wavelet object
                  
    level       : integer
                  decomposition level
                  
    mode        : integer
                  scaling or wavelet coefficient
             
        
    Returns
    --------
    out : numpy-tuple
          (time,reconstruction,signal)
    
    """    
    
    #generate the scaling and wavelet functions
    s1,w1,t2=w.wavefun(level=20)
    #setup 1D interpolating function for scaling and wavelet functions
    wavelet=interpolate.interp1d(t2, w1)
    scaling=interpolate.interp1d(t2, s1)
    
    
    time=[]
    sig=[]
    s1=np.array(s1,float)
    
   
    #compute the dydactic scale
    l=2**level
    
    #find the support
    end1=math.floor(t2[len(t2)-1])
    d=abs(float(len(coeff))*float(l)-len(x))
    d=int(d)
    
    #range over each element of coefficient
    for i in range(len(coeff)-1):     
       
       #define the interpolation points
       t=np.linspace(l*i,l*i+l*end1,l*end1*(len(x)));
       
       t1=np.linspace(0,end1,end1*(len(x))*l);
       
       #multiply the coefficient value with scaling or interpolation function
       if mode==0:
           val=coeff[i]*scaling(t1)
       else:
           val=coeff[i]*wavelet(t1)
           

       ratio=end1
       #compute the translation
       inc=len(t)/ratio
       
       
       if i==0:
               sig=np.append(sig,val)
               time=np.append(time,t)
               
       else:   
               #compute the incremental sum of signals
               v1=val[0:len(t)-inc]
               
               if  inc < len(t):
                   v2=val[len(t)-inc:len(t)]
                   sig[i*inc:i*inc+len(t)]=sig[i*inc:i*inc+len(t)]+v1
                   sig=np.append(sig,v2)
                   time=np.append(time,np.linspace(l*i+l*end1-l,l*i+l*end1,inc))       
               else:
                   sig=np.append(sig,val)
                   time=np.append(time,t)      
    
    #flatten the arrays      
    sig=np.array(sig).flatten()
    time=np.array(time).flatten().ravel()

    #scale the values due to didactic decomposition
    sig=sig/(math.sqrt(2)**level)

    #upsamples the value of signal
    x=np.repeat(x,len(sig)/len(x))

    #return the signals,which can be plotted
    return time,sig,x