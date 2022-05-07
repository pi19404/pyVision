---
layout: post
title:  Fraction Delays using Linear Interpolation and Resampling
category: Signal Processing
---

### Introduction
In this article we look at implementing fractional delays using Linear Interpolation and Resampling Techniques

A integral delay of $M$ is obtained by a delay line of length $M$.A factional  delay is implemented cascading the integral part of delay with a block which can approximate a constant phase delay equal to fractional part of $m$

Fractional delay filters receive a sequential stream of input samples and produce a corresponding sequential stream of interpolated output values.

The most intuitive way of obtaining fractional delay is interpolation	.

For an ideal fractional-delay filter, the frequency response should be equal to that of an ideal delay

$\displaystyle H^\ast(e^{j\omega}) = e^{-j\omega\Delta}$


where  $ \Delta = N+ \eta$ denotes the total desired delay of the filter
Thus, the ideal desired frequency response is a linear phase term corresponding to a delay of $ \Delta$ samples.

### Linear Interpolation
Linear interpolation works by effectively drawing a straight line between two neighboring samples and returning the appropriate point along that line.

More specifically, let $ \eta$ be a number between 0 and 1 which represents how far we want to interpolate a signal $ y$ between time $ n$ and time $ n+1$ . Then we can define the linearly interpolated value  $ \hat y(n+\eta)$ as follows:

 $\displaystyle \hat y(n+\eta) = (1-\eta) \cdot y(n) + \eta \cdot y(n+1) $

$\displaystyle \hat y(n+\eta) = y(n) + \eta\cdot\left[y(n+1) - y(n)\right].$

Thus, the computational complexity of linear interpolation is one multiply and two additions per sample of output.

In case of delay filter $\eta$ is the fractional part of the delay.Thus we pass the sequence throught a filter

for example for a delay of 1/4
$\displaystyle {\hat y}\left(n-\frac{1}{4}\right)
\;=\;\frac{3}{4} \cdot y(n) + \frac{1}{4}\cdot y(n-1) $

The python code for implementing fractional delay by interpolation can be found below

<pre class="brush:python">


def convolution(signal,h):
    """ function that performs linear convolution """
    output=scipy.convolve(signal,h,"same")    
    return output
    
    
def fdelay(signal,N):
    """ function introduces a fractional delay of N samples 
    
    Parameters
    -----------
    signal : numpy-array,
             The input signal
             
    N      : factional
             delay
        
    Returns
    --------
    out : numpy-array
          delayed signal
    
    """    
    f,i=math.modf(N)
    #perform integral delay
    signal=delay(signal,i)

    #perform linear interpolation for fractional delay    
    output=convolution(signal,[f,i-f])

    return output

def delay(signal,N):
    """ function introduces a circular delay of N samples 
    
    Parameters
    -----------
    signal : numpy-array,
             The input signal
             
    N      : integer
             delay
        
    Returns
    --------
    out : numpy-array
          delayed signal
    
    """
    if N==0:
        return signal;
        
    if N >= len(signal):
        N=N-len(signal)
    
    if N <0:
        N=N+len(signal)
    
   
    d=signal[len(signal)-N:len(signal)];#numpy.zeros((1,N+1));    
    signal1=numpy.append(d,signal[0:len(signal)-N])
    return signal1;
    
    </pre>



### Upsampling Technique

Let us assume we can express the fractional delay as rational number $\frac{M}{N}$

The steps to introduce fractional delay are 

 - upsample the sequence by a factor $N$ which simple inserts N zeros between adjacent samples.
 - interpolate the zero values using low pass filter 
 - Delay the signal by $M$ samples
 - downsample by a factor $N$

<pre class="brush:python">

def fdelay(signal,N,mode="upsample"):
    """ function introduces a fractional delay of N samples 
    
    Parameters
    -----------
    signal : numpy-array,
             The input signal
             
    N      : factional
             delay
        
    mode   : "linear " - linear interpolation
             "upsample " - upsampling technique
    Returns
    --------
    out : numpy-array
          delayed signal
    
    """    
 
    if mode=="linear":      
        f,i=math.modf(N)
        #perform integral delay
        signal=delay(signal,i)           
        #perform linear interpolation for fractional delay    
        output=convolution(signal,[f,i-f])
    if mode=="upsample":
        N=math.ceil(N*100)/100
        #get rational approximation
        result=fractions.Fraction(N).limit_denominator(20)
        num=result.numerator;
        den=result.denominator
        #upsample the signal and interpolate

        out1=scipy.signal.resample(signal,den*len(signal))
        #delay the signal
        out1=delay(out1,int(num))        
        #downsample the signal

        out1=scipy.signal.resample(out1,len(signal))
        
        output=out1
         
    return output

</pre>

The Rational number has been chosse such that denominator is limited to 20 ,so that we are not
upscaling by a large factor.

### Code
The code for the same can be found in the pyVision github repository in files

 - [Utils.py](https://github.com/pi19404/pyVision/blob/master/pySignalProc/Utils.py)

The function `fdelay` implements fractional delay while function `delay` implements integer delay.
The `mode` parameter of `fdelay` function specified which method to use to perform fractional delay operations.
presently it support `linear` - Linear Interpolation and `upsample` - Upsamling technique

