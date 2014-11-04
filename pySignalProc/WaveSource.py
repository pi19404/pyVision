# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 21:08:44 2014

@author: pi19404
"""

            
from pyCommon import *


class WaveSource(object):
    def __init__(self,position,attenuation,phase,velocity,carrier,Fs):
        self.position=np.array(position)
        self.attenuation=attenuation
        self.phase=phase
        self.carrier=carrier
        
        #generate default modulated sinusiodal waveform
        to=100;
        t1=100+(100*Fs/3000);
        l=1000*Fs/3000;       
        
        waveform=4*Utils.sinepulse(l,to,t1,carrier,Fs)
        
        self.signal=waveform