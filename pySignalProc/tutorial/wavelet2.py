# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:41:11 2014

@author: pi19404
"""


from pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title


def multiplotSpectrum(y,Fs,text=None,phase=False):
     """
     Plots a  Amplitude Spectrum of y(t)
     """
     
     
     index=len(y)
     levels=index-2
     for i in range(len(y)):
         n = len(y[i]) 
         #
         freq = np.fft.fftshift(np.fft.fftfreq(y[i].size,1.0)) # two sides frequency range
         #freq=freq[range(n/2)] # one side frequency range
         #print freq,y.size
         Y = np.fft.fftshift(numpy.fft.fft(y[i]))/n # fft computing and normalization
         #np.fft.fftshift
         if i==0 or i==1:
             f1=Fs/(2**levels)
         elif i!=len(y)-1:
             f1=Fs/(2**(levels-i+1))
         else:
             f1=Fs
         freq=freq*f1
            
         
         subplot(index,1,i+1)
         
         plot(freq,len(y[i])*abs(Y),'r') # plotting the spectrum
         plt.grid()
         if Fs!=1:
             xlabel('Freq (Hz)')
             ylabel('|F(freq)|')
         else :
             xlabel('W (rad)')
             ylabel('|F(W)|')
         
         if phase==True:
             subplot(index,2,i+1)
             plot(freq,np.angle(Y)/math.pi,'r') # plotting the spectrum
             if Fs!=1:
                 xlabel('Freq (Hz)')
                 ylabel('P(freq)')
             else :
                 xlabel('W (rad)')
                 ylabel('P(W)')
             plt.grid()
         
         #plotSpectrum(y,Fs)
         #show()    
         #return Y,freq
         
def coef_pyramid_plot(coefs, first=0, scale='uniform', ax=None,sig=None,w=None):
    """
    Parameters
    ----------
    coefs : array-like
        Wavelet Coefficients. Expects an iterable in order Cdn, Cdn-1, ...,
        Cd1, Cd0.
    first : int, optional
        The first level to plot.
    scale : str {'uniform', 'level'}, optional
        Scale the coefficients using the same scale or independently by
        level.
    ax : Axes, optional
        Matplotlib Axes instance

    Returns
    -------
    Figure : Matplotlib figure instance
        Either the parent figure of `ax` or a new pyplot.Figure instance if
        `ax` is None.
    """


    c1=[]
    t1=[]
    n_levels = len(coefs)
    
    #n_levels+1
    for i in range(first+1,n_levels+1):
        ax=fig.add_subplot(n_levels+1,1,i)
        #c=coeff[i-1]
        
        
        if i==1:
            rep=2**(n_levels-i)
        elif i==2:
            rep=2**(n_levels-i+1)
        else:
            rep=2**(n_levels-i+1)
        #print rep,n_levels,len(coeff[i-1])
       
        if i==1 :
            #print rep
            c=[]
            t=[]
            for k in range(len(coefs[i-1])):
                for m in range(rep):
                       
                        if rep*k+m > len(sig):
                            bb=1
                            break;
                        bb=0
                        c.append(coefs[i-1][k])
                        t.append((rep*k+m))
                        
                if k!=len(coefs[i-1])-1:
                    c.append(coefs[i-1][k+1])
                    t.append((rep)*k+m)
                    
                if bb==1:
                    break

        else:
            
            c=[]
            t=[]
            for k in range(len(coefs[i-1])):
                for m in range(2*rep):

                        if rep*k+(m+1)/2 > len(sig):
                            bb=1
                            break;                        
                        if m<=(rep-1):
                            c.append(coefs[i-1][k])
                            t.append((rep*k+(m+1)/2))
                            #print rep*k,((m+1)/2),m
                        elif m>=(rep-1):
                            c.append(-coefs[i-1][k])
                            t.append((rep*k+(m+1)/2))                        
           


        print min(t),max(t)
        c=np.array(c)
        t=np.array(t)
        ax.plot(t,c)
        ax.grid()
        c1.append(c)
        t1.append(t)
        
    
        
    ax=fig.add_subplot(n_levels+1,1,i+1)
    ax.plot(range(len(sig)),sig)    
    c1.append(sig)
    t1.append(range(len(sig)))    
    return fig,c1,t1



#generate wavelet object    
w = pywt.Wavelet('haar')
scaling, wavelet, x = w.wavefun()

#plotting the wavelet and scaling functions figures 1,2
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8,6))
ax1, ax2 = axes
ax1.grid()
ax1.plot(x, scaling);
ax1.set_title('Scaling function, N=8');
ax1.set_ylim(-1.2, 1.2);
ax1.set_xlim(-0.2, 1.2);

ax2.set_title('Wavelet, N=8');
ax2.tick_params(labelleft=False);
ax2.plot(x-x.mean(), wavelet);
ax2.grid()
fig.tight_layout()


#plotting the wavelet decomposition coefficients asnd frequency response fig 3,4
print w.dec_lo,w.dec_hi
figure()
s=np.append(np.array(w.dec_lo),np.zeros(1000))
Utils.plotSpectrum(s,2,"Frequeny response",True)


figure()
s=np.append(np.array(w.dec_hi),np.zeros(1022))
Utils.plotSpectrum(s,2,"Frequeny response",True)


#plotting the sinc squared function and frequency response fig 4,5

tmin = -1;
tmax = 1;
fo=40
FS=[240]


t = linspace(tmin, tmax, abs(tmax-tmin)*FS[0]);
x=Utils.squared_sinc(fo,t)

figure()
Utils.plotSpectrum(x,FS[0])


#plottig the frequency response of downsampled signal
x=x[range(0,len(x),2)]
figure()
Utils.plotSpectrum(x,FS[0])


#plotting sine pulse and wavelet coefficients for 3 level decomposition and its
#freuqency response fig 6,7
x=Utils.sinepulse(1024,0,1024,10)

coeff=pywt.wavedec(x,w,level=3,mode='sym')

#plot projection onto wavelet basis
fig, axes = plt.subplots(2, 1, figsize=(9,11), sharex=True)
ax=axes[0]
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes[0],sig=x,w=w) # omit smoothing coefs


#plot frequency response of wavelet coefficients
figure()
coeff.append(x)
multiplotSpectrum(coeff,FS[0])


#plotting sinc squared signal wavelet projection and freq response fig 8,9    
x=Utils.squared_sinc(fo,t)

coeff=pywt.wavedec(x,w,level=3,mode='sym')

#plot projection onto wavelet basis
fig, axes = plt.subplots(2, 1, figsize=(9,11), sharex=True)
ax=axes[0]
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes[0],sig=x,w=w) # omit smoothing coefs


#plot frequency response of wavelet coefficients
figure()
coeff.append(x)
multiplotSpectrum(coeff,FS[0])

#plotting sine pulse and wavelet coeff and frequency response for 5 level decomposition fig 10,11

x=Utils.sinepulse(1024,0,1024,100.0/32)
coeff=pywt.wavedec(x,w,level=5,mode='sym')


#plot projection onto wavelet basis
fig, axes = plt.subplots(1, 1, figsize=(9,9))
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes,sig=x,w=w) # omit smoothing coefs
fig.tight_layout()

#plot frequency response of wavelet coefficients
figure()
coeff.append(x)
multiplotSpectrum(coeff,FS[0])
show()

