# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 00:42:17 2014

@author: pi19404
"""

from pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title


def plot_diff(t1,x1,t2,x2,tit,stem=False):
    n=len(t1)
    m=len(t2)
    
    
    x3=[]
    k=0
    i=0
    for i1 in range(n-1):
        if t2[i] == t2[i+1] and t1[k] != t1[k+1]:
            x3.append(x1[k])
            x3.append(x1[k])                      
            i=i+1
            k=k+1
        else:
            x3.append(x1[k])
            k=k+1
        i=i+1
        

        
    x3.append(x1[k])    
       
    
    x3=np.array(x3)
    fig=figure()
    ax1=fig.add_subplot(1,1,1)
    
    
    ax1.plot(t2,x3,'r')
    if stem==True:
        ax1.stem(t2,x2)
    ax1.fill_between(t2,x3,x2,facecolor='red')
    ax1.set_title(tit)
    ax1.grid()

def func(t):
    return np.sin(2*math.pi*1*t)

def projection(x,t,scale):
    m=len(t)
    div1=scale
    div=(m)/div1
    
    
    
    t=linspace(0,t[m-1],div+1)    
    
    
    t1=[]
    y=[]
    i=0
    i2=0
    for i1 in t:
         
        if i >= m-1:
            #y.append(v)
            #t1.append(tx[k+1])            
            break
        v=np.mean(x[i:i+scale])
        tx=np.linspace(t[i2],t[i2+1],scale+1)
        for k in range(len(tx)):
            y.append(v)
            t1.append(tx[k])     
            if k!=len(tx)-1:
                i=i+1 
        

        i2=i2+1
        
    y=np.array(y) 
    y=y#*math.sqrt(scale)
    t1=np.array(t1)
    return y,t1


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


T=8
t=np.linspace(0,1,257)

t1=t
x=func(t)
y,t=projection(x,t,T)
plot_diff(t1,x,t,y,"picewise constan approximation",True)

#plt.stem(t,y)
#ax1.fill_between(t,y,func(t),facecolor='red')
#e=y-func(t)
#MSE=np.mean(e*e)
#print MSE
#ax1.grid()
#title='MSE',MSE,"T=",T
#ax1.set_title(title)


y1,t2=projection(x,t1,T/2)




plot_diff(t,y,t2,y1,"diff subspace")
#print len(t),len(t),len(y1)
#

x=[1,1,1,1,1,1]
x=np.array(x)

coeff = pywt.wavedec(x,wavelet=w,mode='cpd',level=2)
for i in range(len(coeff)):
    print coeff[i]
show()