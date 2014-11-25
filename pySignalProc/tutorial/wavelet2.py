# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:41:11 2014

@author: pi19404
"""


from pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title


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
                        c.append(coefs[i-1][k])
                        t.append((rep*k+m))
                if k!=len(coefs[i-1])-1:
                    c.append(coefs[i-1][k+1])
                    t.append((rep)*k+m)

        else:
            
            c=[]
            t=[]
            for k in range(len(coefs[i-1])):
                for m in range(2*rep):
#                        if m==0:
#                            c.append(0)
#                            t.append((rep*k+(m+1)/2))
                        
                        if m<=(rep-1):
                            c.append(coefs[i-1][k])
                            t.append((rep*k+(m+1)/2))
                            #print rep*k,((m+1)/2),m
                        elif m>=(rep-1):
                            c.append(-coefs[i-1][k])
                            t.append((rep*k+(m+1)/2))                        
                        
                #c.append(coefs[i-1][k])
                #t.append((rep)*k+(m+1)/2)            
            
        #print t
        #print c
        c=np.array(c)
        t=np.array(t)
        ax.plot(t,c)
        c1.append(c)
        t1.append(t)
        
    
        
    ax=fig.add_subplot(n_levels+1,1,i+1)
    ax.plot(sig)    
    c1.append(sig)
    t1.append(range(len(sig)))    
    return fig,c1,t1
    
w = pywt.Wavelet('haar')
scaling, wavelet, x = w.wavefun()

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

print w.dec_lo,w.dec_hi
figure()
s=np.append(np.array(w.dec_lo),np.zeros(1000))
Utils.plotSpectrum(s,2,"Frequeny response",True)


figure()
s=np.append(np.array(w.dec_hi),np.zeros(1022))
Utils.plotSpectrum(s,2,"Frequeny response",True)


tmin = -1;
tmax = 1;
fo=40
FS=[240]


t = linspace(tmin, tmax, abs(tmax-tmin)*FS[0]);
x=Utils.squared_sinc(fo,t)

figure()
Utils.plotSpectrum(x,FS[0])


x=x[range(0,len(x),2)]
figure()
Utils.plotSpectrum(x,FS[0])


x=Utils.squared_sinc(fo,t)
x=Utils.sinepulse(1000,0,1000,10)

coeff=pywt.wavedec(x,w,level=3,mode='sym')

fig, axes = plt.subplots(2, 1, figsize=(9,11), sharex=True)

ax=axes[0]
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes[0],sig=x,w=w) # omit smoothing coefs



figure()
coeff.append(x)
print len(coeff)
Utils.multiplotSpectrum(coeff,FS[0])
    


show()

