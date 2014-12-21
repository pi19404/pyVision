# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 22:11:42 2014

@author: pi19404
"""

from pySignalProc.pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title
from scipy import interpolate
from tabulate import tabulate
from scipy import integrate

from cvxopt import matrix, solvers
from pySignalProc.wavelet.pywtUtils import *

from pySignalProc.PieceWisePolynomial import *


from scipy.signal import (freqz, butter, bessel, cheby1, cheby2, ellip, 
                              tf2zpk, zpk2tf, lfilter, buttap, bilinear, cheb2ord, cheb2ap
                              )
from numpy import asarray, tan, array, pi, arange, cos, log10, unwrap, angle

from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from collections import defaultdict

    
def zplane(z, p, filename=None):
    """Plot the complex z-plane given zeros and poles.
    """
    
    # get a figure/plot
    ax = plt.subplot(1, 1, 1)
    # TODO: should just inherit whatever subplot it's called in?
 
    # Add unit circle and zero axes    
    unit_circle = patches.Circle((0,0), radius=1, fill=False,
                                 color='black', ls='solid', alpha=0.1)
    ax.add_patch(unit_circle)
    axvline(0, color='0.7')
    axhline(0, color='0.7')
    
    # Plot the poles and set marker properties
    poles = plt.plot(p.real, p.imag, 'x', markersize=9, alpha=0.5)
    
    # Plot the zeros and set marker properties
    zeros = plt.plot(z.real, z.imag,  'o', markersize=9, 
             color='none', alpha=0.5,
             markeredgecolor=poles[0].get_color(), # same color as poles
             )
 
    # Scale axes to fit
    r = 1.5 * np.amax(np.concatenate((abs(z), abs(p), [1])))
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])
#    ticks = [-1, -.5, .5, 1]
#    plt.xticks(ticks)
#    plt.yticks(ticks)
 
    """
    If there are multiple poles or zeros at the same point, put a 
    superscript next to them.
    TODO: can this be made to self-update when zoomed?
    """
    # Finding duplicates by same pixel coordinates (hacky for now):
    poles_xy = ax.transData.transform(np.vstack(poles[0].get_data()).T)
    zeros_xy = ax.transData.transform(np.vstack(zeros[0].get_data()).T)    
 
    # dict keys should be ints for matching, but coords should be floats for 
    # keeping location of text accurate while zooming
 
    # TODO make less hacky, reduce duplication of code
    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in poles_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.iteritems():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )
 
    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in zeros_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.iteritems():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )
 
    if filename is None:
        print ""
    else:
        plt.savefig(filename)
        print 'Pole-zero plot saved to ' + str(filename)
        
        

from math import factorial



def calculate_combinations(p, k):    
    """ calculates the combination p+kCk """
    r=0.5*combination(p+k,k);
    return r

def combination(n,r):    
    """ calculates the combination nCr """
    r=factorial(n) /( factorial(n-r) * factorial(r))
    return r
    
def sinp(p,pad=0):
    """ calculates the coefficients trigonometric polynomial sin^2p(x) """
    
    result=np.zeros(2*p+1)
    g1=1.0/(2**(2*p))
    t1=combination(2*p,p)
    result[p]=g1*t1
    

    for k in range(p):        
        v=g1*combination(2*p,k)
        if (p-k)%2!=0:
            v=-v       
        result[k]=v
        result[2*p-k]=v
    
    l=2*pad-2*p
    result=np.append(np.zeros(l/2),result)    
    result=np.append(result,np.zeros(l/2))
    
    return result
    
  
def unique_real_zeros(z):            
    """ calculates the stable unique zeros from magnitude squared transfer function """
    unique = scipy.unique(z)
    
    counts=np.zeros(len(unique))

    for i in range(len(unique)):
        for k in range(len(z)):
            if unique[i]==z[k]:
                counts[i]=counts[i]+1

    zeros=[]
    
    for e in range(len(unique)):
        if (np.imag(unique[e])==0 or abs(np.imag(unique[e]))<1e-3) and (abs(np.real(unique[e])-1)<1e-3 or abs(np.real(unique[e]))<1):                    
            if counts[e]>1:
                multi=counts[e]/2
                print multi
                zeros.append(np.repeat(np.real(unique[e]),multi))
            if counts[e]==1 and (abs(np.real(unique[e])-1)<1e-3 or abs(np.real(unique[e]))<1):
                zeros.append(np.real(unique[e]))


    zeros=np.array(zeros,float)
    return zeros

def stable_zeros(z):    
    """ get the stable complex zeros from magnitude squared transfer function """
   
    fz=[]
    for l in z:
        if np.imag(l)!=0:
            l1=l*np.conj(l)
            l1=np.abs(l)            
            if l1<1 or abs(l1-1)<1e-3:
                fz.append(l)
      
      
    
    fz=np.array(fz)
    return fz
    
    

def cqf_filters(dec_lo):
    """ computs the decomposition and analysis low and high pass filters given decomposition low pass filter """
    index=np.array(range(len(dec_lo)))
    dec_hi=np.exp(-1j*math.pi*index)*np.flipud(dec_lo)
    dec_hi=dec_hi.real
    
    
    recon_lo=np.exp(-1j*math.pi*index)*dec_hi.real
    recon_lo=recon_lo.real
    
    recon_hi=-np.exp(-1j*math.pi*index)*dec_lo.real
    recon_hi=recon_hi.real
    

    return dec_lo,dec_hi,recon_lo,recon_hi


def trigonometric_filter_zeros(zeros):
    """ function return the zeros of trigonometric polynomial L(w)"""
    v=[]
    z=[]        
    for i in range(zeros):    
        v.append(calculate_combinations(zeros-1,i))
        
    v=np.array(v,float)   
    
    result=np.zeros(2*zeros-1)
    for k in range(zeros):    
        result=result+v[k]*sinp(k,zeros-1)
        
      
    den=np.zeros(len(result))
    den[len(den)-1]=1
   
    #convertize the transfer function to pole zero representation
    z1, p, k = tf2zpk(result,den)

    z=np.append(z,unique_real_zeros(z1))
    z=np.append(z,stable_zeros(z1))

        
    return z
    
def compute_coeffs(z,z1):
    """ function computes the filter coefficients of decomposition low pass filter
        given zeros at pi and zeros of trigonmonetric polynomial 
        for daubach filter """
        
    #adding poles for causal system    
    zeros=len(z)
    p=np.zeros(zeros-1)
    p=np.array(p)
    z=np.append(z,z1)
    
    h = zpk2tf(z,p,[1])   
    h=np.array(h[0])

    #normalizing the filter coefficients    
    h=h/math.sqrt(np.sum(h**2))
    return h,z,p     
    

def DaubechiesWavelet(zeros=2):


    #populating zeros at pi
    z=[]
    for i in range(zeros):
        z.append(-1)


    #compute the zeros corresponding to trigonometric polynomial
    tz=trigonometric_filter_zeros(zeros)

    #compute the low pass filter decomposition filter
    h,z,p=compute_coeffs(z,tz)

    #plotting the pole zero plot
    figure()
    zplane(z, p)

    #computing the wavelet filter coefficients of CQF
    h1,h2,h3,h4=cqf_filters(h)
    
    return h1,h2,h3,h4




dec_lo,dec_hi,recon_lo,recon_hi=DaubechiesWavelet(3)



print dec_lo
print dec_hi
print recon_lo
print recon_hi
    

w=pywt.Wavelet('db3')
print w.rec_lo
print w.rec_hi
print w.dec_lo
print w.dec_hi

figure()
z,p,k=tf2zpk(w.rec_lo,[1]);
zplane(z,p)
show() 
ddddddddd
#print "constrains",np.sum(h1),np.sum(h1**2) 

   




print np.mean((w.rec_lo-h1)**2),"estimation error"


  
show() 
ddddddd 
asdasd        
c=np.convolve(w.dec_lo,np.flipud(w.dec_lo))
print c

z, p, k = tf2zpk(w.dec_lo,[1])
print z,p,k

figure()
zplane(z, p)
figure()
z, p, k = tf2zpk(c,[1])
zplane(z, p)


c=[1,1,2,2]
figure()
z, p, k = tf2zpk(c,[1])
zplane(z, p)

c=np.convolve(c,np.flipud(c))
print c
figure()
z, p, k = tf2zpk(c,[1])
zplane(z, p)

print z

s1=math.sqrt(2)
s2=1.0/s1
z=[s1,s2,-s1,-s2]

h=zpk2tf(z,p,k)
print h,k


show()
dd
dim=2+w.vanishing_moments_psi
dim1=w.vanishing_moments_psi+1
print dim,dim1,w.dec_lo,w.vanishing_moments_psi

A=np.zeros((dim,dim1))
c=np.zeros(dim1)

P=2*np.eye(dim)
P[dim-1,dim-1]=-2
q=np.zeros(dim)
s=np.zeros(dim1)

for i in range(dim1):
       
    if i==0:
        c[i]=math.sqrt(2)
        A[:,i]=np.ones(dim)
        #A[dim-1,i]=0
    else:
        c[i]=0
        d=np.exp(1j*math.pi*np.array(range(0,dim)))
        #d=Utils.upsample(d,2)
        d=d.real
        
        v=d*np.array(range(0,dim))**(i-1)
        A[:,i]=v
        #A[dim-1,i]=0

A=A.T
print A,c
from cvxopt import solvers, matrix, spdiag, log

c1=np.array(w.dec_lo).T



solvers.options['maxiters']=1000

def acent(A, b):
    m, n = A.size
    def F(x=None, z=None):
        if x is None: return 0, matrix(np.random.rand(n,1))
        
        
        
        f = -1*(sum((x**2)))+1
        #print f,"DD",x,z
        Df = 2*(x).T

        if z is None: return f, Df
        H = spdiag(z[0] * x**-2)
        return f, Df, H
    return solvers.cp(F, A=A, b=b)['x']
    
c2=acent(matrix(A),matrix(c))    
c2=np.array(c2)
print c1
c2=c2.reshape(dim)
print c2
print c1/c2
print np.sum(c1**2),np.sum(c2**2),np.sum(c1**1),np.sum(c2**1)

dddd
xx=solvers.coneqp(matrix(P),matrix(q),matrix(A),matrix(c))
print xx['x']




s2=np.sum(c**2)
s1=np.sum(c)
s1=np.sum(c)

print s1,s2