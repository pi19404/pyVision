# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:57:14 2014

@author: pi19404
"""
from pyCommon import *
from numpy import zeros,hstack,matrix,arange,sqrt,eye,fliplr,exp,array
from math import pi
from cvxopt import matrix as matrx 
from cvxopt import solvers
def dftmatrix(N=8): 
    'compute inverse DFT matrices'
    n = arange(N)
    U=matrix( exp(1j*2*pi/N*n*n[:,None] ))/sqrt(N)
    return matrix(U)

def Q_rmatrix(Nf=8):
    'implements the reordering, adding, and stacking of the matrices above'
    Q_r=matrix(hstack([eye(Nf/2),eye(Nf/2)*0])
               +hstack([zeros((Nf/2,1)),fliplr(eye(Nf/2)),zeros((Nf/2,Nf/2-1))]))
    return Q_r

def rearrange_G( x ): 
    'setup to put inequalities matrix with first 1/2 of elements as main variables'
    n=x.shape[0]
    return hstack([x[:,arange(0,n,2)+1], x[:,arange(0,n,2)]])


def L1_min(Phi,y,K):
    # inequalities matrix with 
    M,Nf = Phi.shape
   
   
    G=matrx(rearrange_G(scipy.linalg.block_diag(*[matrix([[-1,-1],[1,-1.0]]),]*Nf) ))
    # objective function row-matrix
    c=matrx(hstack([ones(Nf),zeros(Nf)]))
    # RHS for inequalities
    h = matrx([0.0,]*(Nf*2),(Nf*2,1),'d') 
    # equality constraint matrix
    A = matrx(hstack([Phi*0,Phi]))
    
    # RHS for equality constraints 
    b=matrx(y)
    # suppress standard output
    #old_stdout = sys.stdout
    #sys.stdout = mystdout = StringIO()
    print hstack([Phi*0,Phi]).shape
    sol = solvers.lp(c, G, h,A,b)
    # restore standard output
    #sys.stdout = old_stdout
    sln = array(sol['x']).flatten()[:Nf].round(4)
    return sln
    

# Imports
from sklearn.linear_model import Lasso
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
from matplotlib.pyplot import plot, show, figure, title
import numpy as np


# Initializing constants and signals
#Fs=2000
#l=200
#f=Utils.sinepulse(l,0,l,100,Fs)+Utils.sinepulse(l,0,l,500,Fs)
#t=np.linspace(0,l*1.0/Fs,l);
#t1=t
#N=len(f)
## Displaying the test signal
#f = np.reshape(f, (len(f),1))


# Initializing constants and signals

  

FS = 8e3

f1 = 200
duration = 1./8
t = np.linspace(0, duration*1.0, duration*FS)
N = int(duration*FS)
l=N
f = Utils.sinepulse(l,0,l,f1,FS)
f = np.reshape(f, (len(f),1))

t1=t

#f=Utils.downsample(f,2)
#t=Utils.downsample(t,2)
#plot(t,f,'ro')
M=int(N)/2


k = range(0,N,2)

#k = np.sort(k) # making sure the random samples are monotonic
#k=range(0,N,1)

D =dftmatrix(N)[:M,:]
#Z=np.zeros((N,N))
#D=np.hstack([D,Z])
#D=D.reshape((2*N,N))
#k1=range(0,2*N,1)
#
A = D
#A=np.transpose(A)
#print A.shape,N
b=f[k]

plot(t,f)
plot(t[k],f[k],'go',markersize=5)
#
recons=L1_min(A,b.astype(np.float64),M)
#lasso = Lasso(alpha=0.0001,max_iter=5000)
#lasso.fit(A,b.ravel())
#recons=lasso.coef_.reshape((N,1))
#np.flipud(recons)
recons = dct(recons,axis=0)

plot(t1,recons,'r')
#
#recons_sparse = coo_matrix(lasso.coef_)
#sparsity = 1 - float(recons_sparse.getnnz())/len(lasso.coef_)
#print "sparsity",sparsity

show()




dddd

#tsignal=Utils.addNoise(tsignal,0.00001)
#fsignal,freq=Utils.plotSpectrum(tsignal,2000,"sparse frequency domain signal")


K=2 # components
Nf=128 # number of samples


# setup signal DFT as F
F=zeros((Nf,1))  
F[1]=1
F[2]=0.5
F[Nf-1]=1 # symmetric parts
F[Nf-2]=0.5
ftime = dftmatrix(Nf).H*F # this gives the time-domain signal
ftime = tsignal 


it=1
plt.plot(np.linspace(0,len(tsignal),len(tsignal)),tsignal/np.max(np.abs(tsignal)),'r')
ftime=tsignal[range(0,len(tsignal),it*2)]
ftime1=ftime/np.max(np.abs(ftime))

plt.plot(np.linspace(0,len(tsignal),len(tsignal)/(2*it)),ftime1,'ro',markersize=10)

for i in range(it+1):
    
    Nf=len(ftime)
  
    N=2*Nf
    
    time_samples=np.array(range(0,N/2,2))
    #print time_samples.shape,ftime.shape,N,Nf
    half_indexed_time_samples = (array(time_samples)/2).astype(int)
  
    
    
    Phi = dftmatrix(N/2).real*Q_rmatrix(N)
    Phi_i = Phi[half_indexed_time_samples,:]
    
    # inequalities matrix with 
    G=matrx(rearrange_G(scipy.linalg.block_diag(*[matrix([[-1,-1],[1,-1.0]]),]*N) ))
    # objective function row-matrix
    #c=matrx(hstack([100*ones(N),zeros(N)]))
    #if i==0:
    c=matrx(hstack([ones(N),zeros(N)]))
    #else:
    #c=matrix(hstack(ff,zeros(N)))
    # RHS for inequalities
    h = matrx([0.0,]*(N*2),(N*2,1),'d') 
    # equality constraint matrix
    A = matrx(hstack([Phi_i,Phi_i*0]))
    # RHS for equality constraints 
    b=matrx(ftime[half_indexed_time_samples])
    
    sol = solvers.lp(c, G, h,A,b)
    
    sln = array(sol['x']).flatten()[:N].round(4)
    #ff=np.append(sln,np.flipud(sln)[0:len(sln)])
    ff=sln.reshape((len(sln),1))
    
    ftime1 = dftmatrix(N).H*ff
    ftime1=ftime1.real
    #ftime1=ftime1/np.max(np.abs(ftime1))
    
    ftime=ftime1
    ftime=Utils.downsample(ftime,2);
    ftime=scipy.signal.resample(ftime,2*len(ftime))



tsignal=ftime/np.max(np.abs(ftime))   
plt.plot(np.linspace(0,len(tsignal)/2,len(tsignal)),tsignal,'go')
show() 
#print np.dot((ftime1-ftime).T,(ftime1-ftime))/len(ftime1)
#print np.allclose(ftime1,ftime,1e-5,0.1)

