


from pyCommon import *

from WaveSource import *

class WaveSink(object):
        """
        
        Attributes
        ----------
        postion - position of signal sink/receiver
        noise   - standard deviation of sink/receiver noise
        Fs      - digital sampling frequency
        velocity - velocity of wave
        attenutation - absorbtion loss
        """
    
        def __init__(self,position,Fs,velocity,phase_shift=0,attenuation=1.0,noise=0.1):
            """ Initialization function """
            
            self.position=np.array(position,float)
            self.noise=noise
            self.Fs=Fs
            self.velocity=velocity
            self.attenuation=attenuation
            self.phase_shift=phase_shift

        def propagation_delay(self,source):
            """ the function completes the sample delay of source signal 
            
            Parameters
            -----------
            source - numpy array ,shape (Nxdimension)
                     source location            
            
            Returns
            ----------
            diff - numpy array ,shape (Nx1)
                   distance of receiver to source
            
            """
            diff1=[]
            for i in range(len(source)):
                diff1.append(Utils.distance(self.position,source[i]))
            diff1=np.array(diff1)
            return diff1
            
       
        def propagation_loss(self,source):
            """ the function completes the spreading loss of signal 
            from source to present location
            
            Parameters
            -----------
            source - numpy array ,shape (Nxdimension)
                     source location            
            
            Returns
            ----------
            loss - numpy array ,shape (Nx1)
                   spreading propagation loss
            
            """
            dist=self.propagation_delay(source)
            loss=1.0/dist
            return loss
            
        def sample_delay(self,source):
            """" computes the sample delay of source signal observed at receiver 
            
            Parameters
            -----------
            source - numpy array ,shape (Nxdimension)
                     source location            
            
            Returns
            ----------
            delay - numpy array ,shape (Nx1)
                    sample delay corresponding to propagationtime        
            
            """
            dist=self.propagation_delay(source)
            delay=dist*self.Fs/self.velocity
            return delay
        
        def run(self,sources):
            """" simulates the signal at the wave receiver
            
            Parameters
            -----------
            sources - WaveSource 
                    
            
            Returns
            ----------
            signal - numpy array ,shape (Nx1)
                    wave at the receiver due to multiple sources        
            
            """            
            signals=[]
            for i in range(len(sources)):
                source=(sources[i])
                delay=self.sample_delay(source.position)
                loss=self.propagation_loss(source.position)
                signal=Utils.fdelay(source.signal,delay[0])
                signal=signal*loss[0]*source.attenuation
                signals.append(signal)
            
               
            for i in range(len(sources)):
                signals[i]=Utils.addNoise(signals[i],self.noise)        
                seed=int(np.random.uniform(0,1)*10000);   
                np.random.seed(seed)                 
            
            signals=np.array(signals,float)
             
            
            for i in range(len(sources)):
                source=(sources[i])
                if source.phase!=0:
                    signals[i]=Utils.phase_sift(signals[i],source.phase)
            
            
            signals=np.sum(signals,axis=0)  
            return signals
            



if __name__ == "__main__":  
    
    #souce signal properties
    Fs=48000 
    carrier=1000
    velocity=300

    #souce location
    source=[[5,3],[5,-3],[-5,3],[5,17],[15,3]]
    phase =[0,math.pi,math.pi,math.pi,math.pi]
    attenuation=[1,0.5,0.5,0.5,0.5]
    
    sources=[]
    for i in range(len(source)):
        s=WaveSource(source[i],attenuation[i],phase[i],velocity,carrier,Fs)
        sources.append(s)
    
    
  
    receivers=[10,10]          
    sink=WaveSink(receivers,Fs,velocity,0,1.0,0.001)
    signals=sink.run(sources)    
    
    #print signals
    plt.plot(signals)
    plt.show()
    
    
    
    


            