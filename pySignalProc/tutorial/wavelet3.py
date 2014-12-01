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
            ax.set_title("low pass "+str(n_levels-1))
        elif i==2:
            rep=2**(n_levels-i+1)
            ax.set_title("high pass "+str(n_levels-1))
        else:
            rep=2**(n_levels-i+1)
            ax.set_title("high pass "+str(n_levels-i))
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
        ax.grid()
        ax.plot(t,c)
        
        ax.get_xaxis().set_ticks([])
        #ax.get_yaxis().set_ticks([])        
        c1.append(c)
        t1.append(t)
        
    
        
    ax=fig.add_subplot(n_levels+1,1,i+1)
    ax.plot(range(len(sig)),sig)    
    
    ax.set_title('signal')
    ax.grid()
    ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])          
    c1.append(sig)
    t1.append(range(len(sig)))    
    return fig,c1,t1
    
    
#generate wavelet object    
w = pywt.Wavelet('haar')


### Haar wavelet representing constant function
x=np.array([1,1,1,1,1,1,1,1,1,1],float)

coeff=pywt.wavedec(x,w,level=1)

fig=plt.figure()
fig, axes = fig,fig.axes
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes,sig=x,w=w) # omit smoothing coefs


print w.family_name,"number of vanishing moments ",w.vanishing_moments_psi

### Haar wavelet representing linear function
t=linspace(0,1,100)
x=t
coeff=pywt.wavedec(x,w,level=1)

fig=plt.figure()
fig, axes = fig,fig.axes
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes,sig=x,w=w) # omit smoothing coefs


### Haar wavelet representing quadratic function
t=linspace(0,1,100)
x=t*t
coeff=pywt.wavedec(x,w,level=1)

fig=plt.figure()
fig, axes = fig,fig.axes
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes,sig=x,w=w) # omit smoothing coefs




### Haar wavelet representing quadratic function at high sampling rate
t=linspace(0,1,10000)
x=t*t
coeff=pywt.wavedec(x,w,level=1)

fig=plt.figure()
fig, axes = fig,fig.axes
fig,c1,t1= coef_pyramid_plot(coeff[0:], ax=axes,sig=x,w=w) # omit smoothing coefs


figure()
### Haar wavelet representing quadratic function with 2 level decomposition
t=linspace(0,1,1000)
x=t

x=x**1-x**2+(0.1*x)**3-(2*x)**4
w = pywt.Wavelet('db2')
coeff=pywt.wavedec(x,w,level=1)
x1=x
l=6
print len(coeff)
for i in range(len(coeff)):
    
    plt.subplot(2*l+1,1,i+1)
    plt.plot(coeff[i],'g')
    plt.plot(coeff[i],'ro')
plt.subplot(2*l+1,1,i+2)
plt.plot(x1,'ro')
plt.plot(x,'g')


show()

### db2 wavelet representing quadratic function
figure()
w = pywt.Wavelet('db2')
print w.family_name,"number of vanishing moments ",w.vanishing_moments_psi

t=linspace(0,1,100)
x=t
coeff=pywt.wavedec(x,w,level=1)

for i in range(len(coeff)):
    plt.subplot(3,1,i+1)
    plt.plot(coeff[i],'r')
    plt.plot(coeff[i],'ro')
plt.subplot(3,1,i+2)
plt.plot(x,'ro')
plt.plot(x,'r')


### db2 wavelet representing cubic function
figure()
w = pywt.Wavelet('db2')
print w.family_name,"number of vanishing moments ",w.vanishing_moments_psi

t=linspace(-10,10,100)
x=(3+t)**6

l=1
coeff=pywt.wavedec(x,w,level=1)

for i in range(len(coeff)):
    
    plt.subplot(2*l+1,1,i+1)
    plt.plot(coeff[i],'g')
    plt.plot(coeff[i],'ro')
plt.subplot(2*l+1,1,i+2)
plt.plot(x,'ro')
plt.plot(x,'g')



### db2 wavelet representing higher order polynomial  function with noise
figure()
w = pywt.Wavelet('db4')
print w.family_name,"number of vanishing moments ",w.vanishing_moments_psi

t=linspace(-1,1,100)
x=(t)**6
x1=Utils.addNoise(x,0.1)
l=2
coeff=pywt.wavedec(x1,w,level=2)

for i in range(len(coeff)):
    
    plt.subplot(2*l+1,1,i+1)
    plt.plot(coeff[i],'g')
    plt.plot(coeff[i],'ro')
plt.subplot(2*l+1,1,i+2)
plt.plot(x1,'ro')
plt.plot(x,'g')


show()



figure()
plt.title("scaling function")
for i in range(1,10):
    scaling, wavelet, x = w.wavefun(i)
    plt.subplot(10,1,i)
    plt.plot(x,scaling,'b')
    plt.plot(x,scaling,'ro')

figure()
plt.title("wavelet function")    
for i in range(1,10):
    scaling, wavelet, x = w.wavefun(i)
    plt.subplot(10,1,i)
    plt.plot(x,wavelet,'b')
    plt.plot(x,wavelet,'ro')

    
show()
print w.dec_lo,w.dec_hi

x=np.array(w.dec_lo)
x1=np.flipud(x)

y=np.convolve(x,x1,'valid')


y2=scipy.signal.correlate(x,x)
print y2

n=range(0,len(x))
n=np.array(n)
x2=x*np.exp(1j*math.pi*n)
x2=np.real(x2)
x3=np.flipud(x2)
y3=np.convolve(x2,x3)


y4=np.real(scipy.signal.correlate(x2,x2))
print y4

print y2+y4

Utils.plotSpectrum(np.append(y2+y4,np.zeros(1000)),1)
show()