# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:53:43 2014

@author: pi19404
"""

from pyCommon import *
from matplotlib.patches import Circle


x1 = np.linspace(-0.5,0.5,100)

points=[]
for x in x1:
        x2=0.5-abs(x1)
        points.append([x1,x2])
        points.append([x1,-x2])
points=np.array(points)


E=np.mean(abs(points[:,0])+abs(points[:,1])-0.5)
#print "Error",E
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(x1,(1-x1)/2)
ax.plot(points[:,0],points[:,1],'gs')
ax.plot(0,1.0/2,'rs',markersize=10)
ax.grid()
ax.axis('equal')

from cvxopt import matrix as matrx # don't overrite numpy matrix class
from cvxopt import solvers

from cvxopt import matrix as matrx # don't overrite numpy matrix class
from cvxopt import solvers

from numpy import ones,zeros,hstack,matrix,array

from cStringIO import StringIO
import sys

def rearrange_G( x ): 
    'setup to put inequalities matrix with last 1/2 of elements as main variables'
    n=x.shape[0]
    return hstack([x[:,arange(0,n,2)+1], x[:,arange(0,n,2)]])

    
def L1_min(Phi,y,K):
    # inequalities matrix with 
    M,Nf = Phi.shape
    G=matrx(rearrange_G(scipy.linalg.block_diag(*[matrix([[-1,-1],[1,-1.0]]),]*Nf) ))
    # objective function row-matrix
    c=matrx(np.hstack([ones(Nf),zeros(Nf)]))
    # RHS for inequalities
    h = matrx([0.0,]*(Nf*2),(Nf*2,1),'d') 
    # equality constraint matrix
    A = matrix(hstack([Phi*0,Phi]))
    # RHS for equality constraints 
    b=matrix(y)
    # suppress standard output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    sol = solvers.lp(c, G, h,A,b)
    # restore standard output
    sys.stdout = old_stdout
    sln = array(sol['x']).flatten()[:Nf].round(4)
    return sln
    


def dftmatrix(N=8): 
    'compute inverse DFT matrices'
    n = arange(N)
    U=np.matrix( np.exp(1j*2*pi/N*n*n[:,None] ))
    return np.matrix(U)
    

    
    
x=[1,1,3,3,2,2,1,1,2,2,3,3]
x=np.array(x,float)
#x=scipy.signal.resample(x,len(x)/2)
#x=scipy.signal.resample(x,len(x)*2)
print "original     ",x
x=x[range(0,len(x),2)]
M=len(x)
N=2*len(x)
x=np.array(x,float)

w1=np.fft.fft(x)
w2=dftmatrix(M)
w3=dftmatrix(N)
w=w2
print w.shape
w=w*np.matrix(x).T
#print w1-w.T
#print np.dot((w1-w.T),(w1-w.T).T)
#print np.allclose(w1,w.T)
#dd
w=w.flatten()


w=w.reshape(w.shape[1])


z=np.zeros((N-M,1))
#print x.shape,w.T.shape,z.shape
w=w.T
#print w

w=np.append(w,z,axis=0)
w=w.T
#print w

#print np.real(w)
x=np.fft.ifft(w)*N/M
print "reconstructed",np.real(x)

sol=L1_min(w3.real,w.real.astype(np.float64),N)

asd
Fs=5000
carrier=2000

#generate default modulated sinusiodal waveform
to=100;
t1=100+(100*Fs/3000);
l=1000*Fs/3000;       

waveform=4*Utils.sinepulse(l,to,t1,carrier,Fs)
plt.figure(2)
Utils.plotSpectrum(waveform,Fs)
#plt.plot(waveform)
show()


